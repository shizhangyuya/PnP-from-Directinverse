import json
import argparse
import os
import numpy as np
import tqdm
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_editing_file', type=str, default="data/image700_source2edit_prompt.json")
    parser.add_argument('--metrics', nargs='+', type=str, default=["clip_similarity_target_image"])
    parser.add_argument('--src_image_folder', type=str, default="data/annotation_images")
    parser.add_argument('--eval_image_folder', type=str, default="output/eval20")
    parser.add_argument('--tgt_methods', nargs='+', type=str, default=["instruct-pix2pix",'blended-latent-diffusion','directinversion+masactrl','directinversion+p2p','directinversion+pnp'])
    parser.add_argument('--eval_style_list', nargs='+', type=str, default=["genre","artist","style"])
    parser.add_argument('--result_path', type=str, default="evaluation_result")
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    prompt_editing_file = args.prompt_editing_file
    metrics = args.metrics
    src_image_folder = args.src_image_folder
    eval_image_folder=args.eval_image_folder
    tgt_methods = args.tgt_methods
    eval_style_list=args.eval_style_list

    tgt_image_folders = {}

    result_path = args.result_path

    metrics_calculator = MetricsCalculator(args.device)


    with open(prompt_editing_file, "r") as f:
        prompt_editing_file = json.load(f)

    genre_list = []
    artist_list=[]
    style_list=[]
    score_dict={
        'genre':genre_list,
        'artist':artist_list,
        'style':style_list
    }

    for edit_method in tgt_methods:
        image_count=0
        for file in os.listdir(f'{eval_image_folder}/{edit_method}/genre'):
            image_count += 1

        for key, item in tqdm.tqdm(prompt_editing_file.items(),desc=f"Evaluation of {edit_method}:"):

            #print(f"evaluating image {key} ...")

            image_dict={}
            original_prompt = item["source_prompt"]
            editing_prompt=prompt_make(original_prompt,item['genre_class'],item['artist_class'],item['style_class'])
            editing_list= [editing_prompt["genre_prompt"],editing_prompt["artist_prompt"],editing_prompt["style_prompt"]]

            image_dict['source_prompt']=original_prompt
            image_dict['genre_prompt'] = editing_prompt['genre_prompt']
            image_dict['artist_prompt'] = editing_prompt['artist_prompt']
            image_dict['style_prompt'] = editing_prompt['style_prompt']
            image_dict['genre_edited_image_path'] = os.path.join(eval_image_folder,edit_method,'genre',key+'.jpg')
            image_dict['artist_edited_image_path'] = os.path.join(eval_image_folder,edit_method,'artist',key+'.jpg')
            image_dict['style_edited_image_path'] = os.path.join(eval_image_folder,edit_method,'style',key+'.jpg')

            edit_method_path = result_path+f'/{edit_method}'
            if not os.path.exists(edit_method_path):
                os.makedirs(edit_method_path)
            for style_type in eval_style_list:
                eval_style_path=os.path.join(edit_method_path,f'{style_type}_class_evaluation_result.csv')
                evaluation_result=[key+'.jpg']

                tgt_image_path = image_dict[style_type+'_edited_image_path']

                #print(f"evluating method: {edit_method} in {style_type} type")

                if image_count==0 and style_type=='genre':
                    for style_type in eval_style_list:
                        with open(f'{result_path}/{edit_method}/{style_type}_class_evaluation_result.csv', 'a+',
                                  newline="") as f:
                            csv_write = csv.writer(f)
                            row = [
                                f'Average {style_type} clip-similarity score of {len(score_dict[style_type])} images edited by {edit_method}:{sum(score_dict[style_type]) / len(score_dict[style_type])}']
                            csv_write.writerow(row)
                            score_dict[style_type].clear()

                    break

                tgt_prompt=image_dict[style_type+'_prompt']
                tgt_image = Image.open(tgt_image_path)
                if tgt_image.size[0] != tgt_image.size[1]:
                    # to evaluate editing
                    tgt_image = tgt_image.crop(
                        (tgt_image.size[0] - 512, tgt_image.size[1] - 512, tgt_image.size[0], tgt_image.size[1]))
                    # tgt_image.show()
                    # to evaluate reconstruction
                    # tgt_image = tgt_image.crop((tgt_image.size[0]-512*2,tgt_image.size[1]-512,tgt_image.size[0]-512,tgt_image.size[1]))

                for metric in metrics:
                    #print(f"evluating metric: {metric}")
                    clip_score=calculate_metric(metrics_calculator, tgt_image, tgt_prompt)
                    score_dict[style_type].append(clip_score)
                    evaluation_result.append(clip_score)

                with open(eval_style_path, 'a+', newline="") as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(evaluation_result)

            if image_count==0:
                break
            image_count-=1


