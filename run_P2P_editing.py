import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

from models.p2p_editor import P2PEditor
from prompt_maker import prompt_make

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


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")  # rerun existing images
    parser.add_argument('--data_path', type=str, default="data")  # the editing category that needed to run
    parser.add_argument('--output_path', type=str,
                        default="output")  # the editing category that needed to run
    parser.add_argument('--edit_style_list', type=str,
                        default=['genre','artist','style'])  # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs='+', type=str,
                        default=["0", "1", "2", "3", "4", "5", "6", "7", "8","9"])  # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                        default=["directinversion+p2p"])  # the editing methods that needed to run
    
    parser.add_argument('--total_frame_num', type=int,default=4) 
    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_style_list=args.edit_style_list
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list
    edit_method=edit_method_list[0]

    total_frame_num=args.total_frame_num

    p2p_editor = P2PEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                           num_ddim_steps=50)

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
                edited_image = p2p_editor(edit_method,
                                          image_path=image_path,
                                          prompt_src=original_prompt,
                                          prompt_tar=editing_prompt,
                                          guidance_scale=7.5,
                                          cross_replace_steps=0.4,
                                          self_replace_steps=0.6,
                                          blend_word= None,
                                          eq_params=None,
                                          proximal="l0",
                                          quantile=0.75,
                                          use_inversion_guidance=True,
                                          recon_lr=1,
                                          recon_t=400,
                                          total_frame_num=total_frame_num
                                          )
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)

                print(f"finsh editing image [{image_path}] with [{edit_method}] in {edit_style_list[style_type]} type")

            else:
                print(f"skip image [{image_path}] with [{edit_method}]")

