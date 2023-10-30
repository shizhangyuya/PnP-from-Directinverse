import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
from matrics_calculator import MetricsCalculator
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


def calculate_metric(metrics_calculator, tgt_image,tgt_prompt):
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, None)


all_tgt_image_folders = {
    # results of comparing inversion
    # ---
    "ddim+p2p": "output/ddim+p2p",
    "null-text-inversion+p2p": "output/null-text-inversion+p2p_a800",
    "negative-prompt-inversion+p2p": "output/negative-prompt-inversion+p2p",
    "stylediffusion+p2p": "output/stylediffusion+p2p",
    "directinversion+p2p": "output/directinversion+p2p",
    # ---
    "ddim+masactrl": "output/ddim+masactrl/annotation_images",
    "directinversion+masactrl": "output/directinversion+masactrl",
    # ---
    "1_ddim+pix2pix-zero": "output/ddim+pix2pix-zero/annotation_images",
    "1_directinversion+pix2pix-zero": "output/directinversion+pix2pix-zero/annotation_images",
    # ---
    "1_ddim+pnp": "output/ddim+pnp/annotation_images",
    "1_directinversion+pnp": "output/directinversion+pnp/annotation_images",

}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_editing_file', type=str, default="../data/image700_source2edit_prompt.json")
    parser.add_argument('--metrics', nargs='+', type=str, default=["clip_similarity_target_image"])
    parser.add_argument('--src_image_folder', type=str, default="../data/annotation_images")
    parser.add_argument('--eval_image_folder', type=str, default="output")
    parser.add_argument('--tgt_methods', nargs='+', type=str, default=["directinversion+p2p"])
    parser.add_argument('--eval_style_list', nargs='+', type=str, default=["genre","artist","style"])
    parser.add_argument('--result_path', type=str, default="evaluation_result")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--evaluate_whole_table', action="store_true")  # rerun existing images

    args = parser.parse_args()

    prompt_editing_file = args.prompt_editing_file
    metrics = args.metrics
    src_image_folder = args.src_image_folder
    eval_image_folder=args.eval_image_folder
    tgt_methods = args.tgt_methods
    eval_style_list=args.eval_style_list
    evaluate_whole_table = args.evaluate_whole_table

    tgt_image_folders = {}

    if evaluate_whole_table:
        for key in all_tgt_image_folders:
            if key[0] in tgt_methods:
                tgt_image_folders[key] = all_tgt_image_folders[key]
    else:
        for key in tgt_methods:
            tgt_image_folders[key] = all_tgt_image_folders[key]

    result_path = args.result_path

    metrics_calculator = MetricsCalculator(args.device)



    with open(prompt_editing_file, "r") as f:
        prompt_editing_file = json.load(f)

    for key, item in prompt_editing_file.items():
        print(f"evaluating image {key} ...")

        image_dict={}
        original_prompt = item["source_prompt"]
        editing_prompt=prompt_make(original_prompt,item['genre_class'],item['artist_class'],item['style_class'])
        editing_list= [editing_prompt["genre_prompt"],editing_prompt["artist_prompt"],editing_prompt["style_prompt"]]

        image_dict['source_prompt']=original_prompt
        image_dict['genre_prompt'] = editing_prompt[0]
        image_dict['artist_prompt'] = editing_prompt[1]
        image_dict['style_prompt'] = editing_prompt[2]
        image_dict['genre_edited_image_path'] = os.path.join(eval_image_folder,tgt_methods,'genre',key+'.jpg')
        image_dict['artist_edited_image_path'] = os.path.join(eval_image_folder,tgt_methods,'artist',key+'.jpg')
        image_dict['style_edited_image_path'] = os.path.join(eval_image_folder,tgt_methods,'style',key+'.jpg')


        for edit_method in tgt_methods:
            edit_method_path = result_path+f'/{edit_method}'
            if not os.path.exists(edit_method_path):
                os.makedirs(edit_method_path)
            for style_type in eval_style_list:
                eval_style_path=os.path.join(edit_method_path,f'{style_type}_class_evaluation_result.csv')
                evaluation_result=[key]
                # with open(eval_style_path, 'w', newline="") as f:
                #     csv_write = csv.writer(f)
                #
                #     csv_head = []
                #     for tgt_image_folder_key, _ in tgt_image_folders.items():
                #         for metric in metrics:
                #             csv_head.append(f"{tgt_image_folder_key}|{metric}")
                #
                #     data_row = ["file_id"] + csv_head
                #     csv_write.writerow(data_row)

                tgt_image_path = image_dict[style_type+'_edited_image_path']

                assert os.path.exists(tgt_image_path)
                print(f"evluating method: {edit_method} in {style_type} type")

                tgt_prompt=image_dict[style_type+'_prompt']
                tgt_image = Image.open(tgt_image_path)
                if tgt_image.size[0] != tgt_image.size[1]:
                    # to evaluate editing
                    tgt_image = tgt_image.crop(
                        (tgt_image.size[0] - 512, tgt_image.size[1] - 512, tgt_image.size[0], tgt_image.size[1]))
                    tgt_image.show()
                    # to evaluate reconstruction
                    # tgt_image = tgt_image.crop((tgt_image.size[0]-512*2,tgt_image.size[1]-512,tgt_image.size[0]-512,tgt_image.size[1]))

                for metric in metrics:
                    print(f"evluating metric: {metric}")
                    evaluation_result.append(calculate_metric(metrics_calculator, tgt_image, tgt_prompt))

                with open(eval_style_path, 'a+', newline="") as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(evaluation_result)

