import torch
import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as T
import imageio

from utils.utils import txt_draw, load_512, latent2image
from prompt_maker import prompt_make

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

NUM_DDIM_STEPS = 50


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list = [latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list = []
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps
                latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(cond, latent)
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        return inverted_x, rgb_reconstruction, latent_reconstruction


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)


def register_attention_control_efficient(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            clip_length=int(batch_size/3)
            h = self.heads

            #for spatio_temp_self_attention

            with_self=True
            with_prev=False
            with_next=False
            Spatio_temp_list=[]

            first=0
            last=clip_length-1
            middle=int(clip_length/2)

            for i in spatio_temp_list:
                if i == 'first':
                    Spatio_temp_list.append(first)
                elif i == 'middle':
                    Spatio_temp_list.append(middle)
                elif i == 'last':
                    Spatio_temp_list.append(last)
                elif i == 'prev':
                    with_prev=True
                elif i == 'next':
                    with_next==True
                else:
                    with_self=True

            self_list=list(range(batch_size))
            prev_list=[0]+self_list[:clip_length-1]+[clip_length]+self_list[clip_length:clip_length*2-1]+[clip_length*2]+self_list[clip_length*2:clip_length*3-1]
            next_list=self_list[1:clip_length]+[clip_length-1]+self_list[clip_length+1:clip_length*2]+[clip_length*2-1]+self_list[clip_length*2+1:clip_length*3]+[clip_length*3-1]
            
            

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            if not is_cross and self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000):
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)

                source_batch_size = int(q.shape[0] // 3)
                # inject unconditional
                q[source_batch_size:2 * source_batch_size] = q[:source_batch_size]
                k[source_batch_size:2 * source_batch_size] = k[:source_batch_size]
                # inject conditional
                q[2 * source_batch_size:] = q[:source_batch_size]
                k[2 * source_batch_size:] = k[:source_batch_size]

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
            else:
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                if batch_size > 3:
                    # k = k[[2]*4+[6]*4+[10]*4, :]

                    k_list=[]
                    if with_prev:
                        k_list.append(k[prev_list,:])
                    if with_self:
                        k_list.append(k[self_list,:])
                    for frame in Spatio_temp_list:
                        index_list = []
                        for i in range(3):
                            frame_index = []
                            frame_index += [frame + i * clip_length] * clip_length
                            index_list.append(frame_index)
                                          
                        k_list.append(torch.cat([k[frame_index, :] for frame_index in index_list],dim=0))

                    if with_next:
                        k_list.append(k[next_list,:])

                    k=(torch.cat(k_list, dim=1))

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)

            v = self.to_v(encoder_hidden_states)
            if batch_size>3:
                # v = v[[2]*4+[6]*4+[10]*4, :]

                v_list = []
                if with_prev:
                    v_list.append(v[prev_list,:])
                if with_self:
                    v_list.append(v[self_list,:])
                for frame in Spatio_temp_list:
                    index_list = []
                    for i in range(3):
                        frame_index = []
                        frame_index += [frame + i * clip_length] * clip_length
                        index_list.append(frame_index)

                    v_list.append(torch.cat([v[frame_index, :] for frame_index in index_list], dim=0))

                if with_next:
                        v_list.append(v[next_list,:])

                v = (torch.cat(v_list, dim=1))

            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)

            return to_out(out)

        return forward

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1,
                                             2]}  # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)



def register_conv_control_efficient(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states) #2560 to 1280

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)


class PNP(nn.Module):
    def __init__(self, model_key, n_timesteps=NUM_DDIM_STEPS, device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps = NUM_DDIM_STEPS
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self, image_path):
        # load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t, guidance_scale, noisy_latent,frame_num):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent] + [x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds.repeat(frame_num,1,1), self.text_embeds.repeat(frame_num,1,1)], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self, image_path, noisy_latent, target_prompt, guidance_scale=7.5, pnp_f_t=0.8, pnp_attn_t=0.5,frame_num=None):
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps, guidance_scale, noisy_latent,frame_num=frame_num)

        return edited_img

    def sample_loop(self, x, guidance_scale, noisy_latent,frame_num):
        interpolation_timestep=40
        total_frame_num=frame_num
        current_frame_num=1
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in tqdm.tqdm(enumerate(self.scheduler.timesteps), desc="Sampling"):

                source_latents = noisy_latent[-i - 1]
                if i == interpolation_timestep:

                    target_latents_list = torch.split(x.repeat(total_frame_num,1,1,1), 1, dim=0)
                    new_latents = []
                    for index, frame in enumerate(range(total_frame_num)):
                        source_rate = 1.0 * (total_frame_num - frame - 1) / (total_frame_num - 1.)
                        target_latents = target_latents_list[index]
                        edited_latents = source_rate * source_latents + (1.0 - source_rate) * target_latents
                        new_latents.append(edited_latents)

                    current_frame_num=total_frame_num
                    x = torch.cat(new_latents, dim=0)

                if gradually and i > interpolation_timestep:
                    target_latents_list = torch.split(x, 1, dim=0)
                    new_latents = []
                    for index, frame in enumerate(range(total_frame_num)):
                        source_rate = (1.0 * (total_frame_num - frame - 1) / (total_frame_num - 1.))/(50-interpolation_timestep)
                        target_latents = target_latents_list[index]
                        edited_latents = source_rate * source_latents + (1.0 - source_rate) * target_latents
                        new_latents.append(edited_latents)

                    x = torch.cat(new_latents, dim=0)

                source_latents=source_latents.repeat(current_frame_num,1,1,1)
                x = self.denoise_step(x, t, guidance_scale, source_latents,frame_num=current_frame_num)

            decoded_latent = self.decode_latent(x)

        return decoded_latent


model_key = "models/stable-diffusion-v1-5"
toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
toy_scheduler.set_timesteps(NUM_DDIM_STEPS)

timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=NUM_DDIM_STEPS,
                                                       strength=1.0,
                                                       device=device)
model = Preprocess(device, model_key=model_key)
pnp = PNP(model_key)


def edit_image_ddim_PnP(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                                                         num_steps=NUM_DDIM_STEPS,
                                                                         inversion_prompt=prompt_src)

    edited_image = pnp.run_pnp(image_path, latent_reconstruction, prompt_tar, guidance_scale)

    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255 * np.array(rgb_reconstruction[0].permute(1, 2, 0).cpu().detach())),
        np.uint8(255 * np.array(edited_image[0].permute(1, 2, 0).cpu().detach())),
    ), 1))


def edit_image_directinversion_PnP(
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        image_shape=[512, 512],
        frame_num=None
):
    
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, rgb_reconstruction, _ = model.extract_latents(data_path=image_path,
                                                              num_steps=NUM_DDIM_STEPS,
                                                              inversion_prompt=prompt_src)

    edited_image = pnp.run_pnp(image_path, inverted_x, prompt_tar, guidance_scale,frame_num=frame_num)

    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    current_frame_num=frame_num
    image_list=[]
    for i in range(current_frame_num):
        image_list.append(Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255 * np.array(edited_image[i].permute(1, 2, 0).cpu().detach())),
    ), 1)))



    return image_list

def visualize(image_list,save_path):
    frame_num=len(image_list)
    present_image_save_path=save_path
    edited_image=image_list
    duration=int(1500/frame_num)

    gif_path = os.path.join(present_image_save_path, f'{key}_compared.gif')
    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))
    imageio.mimsave(gif_path, image_list, duration=duration,loop=0)

    for i in range(frame_num):
        image_save_path = os.path.join(present_image_save_path, f'compared_{i}.png')
        if not os.path.exists(os.path.dirname(image_save_path)):
            os.makedirs(os.path.dirname(image_save_path))
        edited_image[i].save(image_save_path)

    for i in range(frame_num):
        image_save_path = os.path.join(present_image_save_path, f'edit_{i}.png')
        if not os.path.exists(os.path.dirname(image_save_path)):
            os.makedirs(os.path.dirname(image_save_path))
        edited_image[i]=edited_image[i].crop(
            (edited_image[i].size[0] - 512, edited_image[i].size[1] - 512, edited_image[i].size[0], edited_image[i].size[1]))
        edited_image[i].save(image_save_path)

    gif_path = os.path.join(present_image_save_path, f'{key}.gif')
    if not os.path.exists(os.path.dirname(gif_path)):
        os.makedirs(os.path.dirname(gif_path))
    imageio.mimsave(gif_path, edited_image, duration=duration,loop=0)

    # 创建一张新的大图
    result_width = 512 * frame_num  # 总宽度为四张图片的宽度之和
    result_height = 512  # 图片高度
    result_image = Image.new("RGB", (result_width, result_height))

    # 将每个小图粘贴到大图中
    x_offset = 0
    for img in edited_image:
        result_image.paste(img, (x_offset, 0))
        x_offset += 512  # 每张图片的宽度

    frame_path = os.path.join(present_image_save_path, f'{key}_frames.png')
    if not os.path.exists(os.path.dirname(frame_path)):
        os.makedirs(os.path.dirname(frame_path))
    result_image.save(frame_path)

def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


image_save_paths = {
    "ddim+pnp": "ddim+pnp",
    "directinversion+pnp": "directinversion+pnp"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")  # rerun existing images
    parser.add_argument('--data_path', type=str, default="data")  # the editing category that needed to run
    parser.add_argument('--output_path', type=str,
                        default="output")  # the editing category that needed to run
    parser.add_argument('--edit_style_list', type=str,
                        default=['genre', 'artist', 'style'])  # the editing category that needed to run
    parser.add_argument('--frame_num', type=int,
                        default=4)  # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                        default=["directinversion+pnp"])  # the editing methods that needed to run
    parser.add_argument('--spatio_temp_frame',nargs='+',type=str,default=['prev','next'])
    parser.add_argument('--gradually',action="store_true")
    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_style_list = args.edit_style_list
    edit_method_list = args.edit_method_list
    edit_method = edit_method_list[0]
    frame_num=args.frame_num
    gradually=args.gradually

    spatio_temp_list=args.spatio_temp_frame

    with open(f"{data_path}/image700_source2edit_prompt.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        original_prompt = item["source_prompt"]
        editing_prompt = prompt_make(original_prompt, item['genre_class'], item['artist_class'], item['style_class'])
        editing_list = [editing_prompt["genre_prompt"], editing_prompt["artist_prompt"], editing_prompt["style_prompt"]]

        for style_type, editing_prompt in enumerate(editing_list):

            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

            present_image_save_path = os.path.join(output_path, edit_method, edit_style_list[style_type], f'{key}')
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"start editing image [{image_path}] with [{edit_method}] in {edit_style_list[style_type]} type")
                setup_seed()
                torch.cuda.empty_cache()

                edited_image = edit_image_directinversion_PnP(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    frame_num=frame_num
                )

                visualize(edited_image,present_image_save_path)

                print(f"finish")

            else:
                print(f"skip image [{image_path}] with [{edit_method}]")


