import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import json
from PIL import Image

from lavis.models import load_model_and_preprocess
from models.pix2pix_zero.ddim_inv import DDIMInversion
from models.pix2pix_zero.scheduler import DDIMInverseScheduler
from models.pix2pix_zero.edit_directions import construct_direction
from models.pix2pix_zero.edit_pipeline import EditingPipeline
from utils.utils import txt_draw

from diffusers import DDIMScheduler
from prompt_maker import prompt_make

NUM_DDIM_STEPS = 50
XA_GUIDANCE = 0.1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

# load the BLIP model
model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption",
                                                          model_type="base_coco",
                                                          is_eval=True,
                                                          device=torch.device(device))

# make the DDIM inversion pipeline
pipe = DDIMInversion.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.num_inference_steps = NUM_DDIM_STEPS

edit_pipe = EditingPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
edit_pipe.scheduler = DDIMScheduler.from_config(edit_pipe.scheduler.config)
edit_pipe.scheduler.num_inference_steps = NUM_DDIM_STEPS


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## convert sentences to sentence embeddings
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device=device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                sent,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concat(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)


def edit_image_ddim_pix2pix_zero(image_path,
                                 prompt_src,
                                 prompt_tar,
                                 guidance_scale=7.5,
                                 image_size=[512, 512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    prompt_str = model_blip.generate({"image": vis_processors["eval"](image_gt).unsqueeze(0).to(device)})[0]
    latent_list, x_inv_image, x_dec_img = pipe(
        prompt_str,
        guidance_scale=1,
        num_inversion_steps=NUM_DDIM_STEPS,
        img=image_gt
    )

    inversion_latent = latent_list[-1].detach()

    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)

    rec_pil, edit_pil = edit_pipe(prompt_str,
                                  num_inference_steps=NUM_DDIM_STEPS,
                                  x_in=inversion_latent,
                                  edit_dir=(mean_emb_tar.mean(0) - mean_emb_src.mean(0)).unsqueeze(0),
                                  guidance_amount=XA_GUIDANCE,
                                  guidance_scale=guidance_scale,
                                  negative_prompt=prompt_str  # use the unedited prompt for the negative prompt
                                  )

    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    out_image = np.concatenate(
        (np.array(image_instruct), np.array(image_gt), np.array(rec_pil[0]), np.array(edit_pil[0])), 1)

    return Image.fromarray(out_image)

def edit_image_directinversion_pix2pix_zero(image_path,
                                            prompt_src,
                                            prompt_tar,
                                            guidance_scale=7.5,
                                            image_size=[512, 512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    # prompt_str = model_blip.generate({"image": vis_processors["eval"](image_gt).unsqueeze(0).to(device)})[0]
    prompt_str=prompt_src
    latent_list, x_inv_image, x_dec_img = pipe(
        prompt_str,
        guidance_scale=1,
        num_inversion_steps=NUM_DDIM_STEPS,
        img=image_gt
    )

    inversion_latent = latent_list[-1].detach()

    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)

    rec_pil, edit_pil = edit_pipe(prompt_str,
                                  num_inference_steps=NUM_DDIM_STEPS,
                                  x_in=inversion_latent,
                                  edit_dir=(mean_emb_tar.mean(0) - mean_emb_src.mean(0)).unsqueeze(0),
                                  guidance_amount=XA_GUIDANCE,
                                  guidance_scale=guidance_scale,
                                  negative_prompt=prompt_str,  # use the unedited prompt for the negative prompt
                                  latent_list=latent_list
                                  )

    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    out_image = np.concatenate(
        (np.array(image_instruct), np.array(image_gt), np.array(rec_pil[0]), np.array(edit_pil[0])), 1)

    return Image.fromarray(out_image)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")  # rerun existing images
    parser.add_argument('--data_path', type=str, default="data")  # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output")  # the editing category that needed to run
    parser.add_argument('--edit_style_list', type=str,
                        default=['genre', 'artist', 'style'])  # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str, default=["ddim+pix2pix-zero",
                                                                            "directinversion+pix2pix-zero"])  # the editing methods that needed to run
    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_style_list = args.edit_style_list

    edit_method_list = args.edit_method_list
    edit_method = edit_method_list[1]

    with open(f"{data_path}/image700_source2edit_prompt.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        original_prompt = item["source_prompt"]
        editing_prompt = prompt_make(original_prompt, item['genre_class'], item['artist_class'], item['style_class'])
        editing_list = [editing_prompt["genre_prompt"], editing_prompt["artist_prompt"], editing_prompt["style_prompt"]]

        for style_type, editing_prompt in enumerate(editing_list):

            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

            present_image_save_path = os.path.join(output_path, edit_method, edit_style_list[style_type], f'{key}.jpg')
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"start editing image [{image_path}] with [{edit_method}] in {edit_style_list[style_type]} type")
                setup_seed()
                torch.cuda.empty_cache()

                edited_image = edit_image_directinversion_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )

                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)

                print(f"finish")

            else:
                print(f"skip image [{image_path}] with [{edit_method}]")


