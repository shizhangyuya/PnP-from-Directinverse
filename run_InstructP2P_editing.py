from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import json
import os
import argparse
from utils.utils import txt_draw
from ldm.util import default, instantiate_from_config
from prompt_maker import prompt_make

sys.path.append("models/instructpix2pix/stable_diffusion")


# from ldm.util import instantiate_from_config

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


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


def edit_instruct_pix2pix(
        edit_method,
        input,
        edit,
        resolution=512,
        steps=50,
        cfg_text=7.5,
        cfg_image=1.5,
):
    if edit_method == "instruct-pix2pix":
        model_wrap = K.external.CompVisDenoiser(model)
        model_wrap_cfg = CFGDenoiser(model_wrap)
        null_token = model.get_learned_conditioning([""])

        input_image_numpy = Image.open(input).convert("RGB")
        width, height = input_image_numpy.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image_numpy, (width, height), method=Image.Resampling.LANCZOS)

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }
            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        image_instruct = txt_draw(f"edit prompt: {edit}")

        return Image.fromarray(
            np.concatenate((image_instruct, input_image_numpy, np.zeros_like(image_instruct), np.array(edited_image)),
                           axis=1))

    else:
        raise NotImplementedError(f"No edit method named {edit_method}")


image_save_paths = {
    "instruct-pix2pix": "instruct-pix2pix",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")  # rerun existing images
    parser.add_argument('--data_path', type=str, default="data")  # the editing category that needed to run
    parser.add_argument('--output_path', type=str,
                        default="output")  # the editing category that needed to run
    parser.add_argument('--edit_style_list', type=str,
                        default=['genre', 'artist', 'style'])  # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                        default=["instruct-pix2pix"])  # the editing methods that needed to run
    parser.add_argument('--checkpoint', type=str,
                        default="models/instruct-pix2pix-00-22000.ckpt")  # the editing methods that needed to run
    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_style_list = args.edit_style_list
    edit_method = args.edit_method_list[0]
    checkpoint = args.checkpoint

    with open(f"{data_path}/image700_source2edit_prompt.json", "r") as f:
        editing_instruction = json.load(f)

    config = OmegaConf.load("models/instructpix2pix/configs/generate.yaml")
    model = load_model_from_config(config, checkpoint, None)
    model.eval().cuda()

    for key, item in editing_instruction.items():

        original_prompt = item["source_prompt"]
        editing_prompt = prompt_make(original_prompt, item['genre_class'], item['artist_class'], item['style_class'],instruct=True)
        editing_list = [editing_prompt["genre_prompt"], editing_prompt["artist_prompt"], editing_prompt["style_prompt"]]

        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

        for style_type, editing_instruction in enumerate(editing_list):

            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

            present_image_save_path = os.path.join(output_path, edit_method, edit_style_list[style_type], f'{key}.jpg')
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"start editing image [{image_path}] with [{edit_method}] in {edit_style_list[style_type]} type")
                setup_seed()
                torch.cuda.empty_cache()
                edited_image = edit_instruct_pix2pix(
                    edit_method=edit_method,
                    input=image_path,
                    edit=editing_instruction
                )

                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)

                print(f"finish")

            else:
                print(f"skip image [{image_path}] with [{edit_method}]")

