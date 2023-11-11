import json
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import random
import os



data_path="data"
output_path="output"
edit_category_list=["0","1","2","3","4","5","6","7","8","9"]

img_prompt_dict={}

genre_dict = {}
artist_dict={}
style_dict={}

genre_list=[]
artist_list=[]
style_list=[]

# 打开文本文件并读取内容
with open('wikiart/genre_class.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        genre_dict[parts[1]] = 0
        genre_list.append(parts[1])

with open('wikiart/artist_class.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        artist_dict[parts[1]] = 0
        artist_list.append(parts[1])

with open('wikiart/style_class.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        style_dict[parts[1]] = 0
        style_list.append(parts[1])

# print(genre_list)
# print(genre_dict)

with open(f"{data_path}/mapping_file.json", "r") as f:
    editing_instruction = json.load(f)

for key, item in editing_instruction.items():

    if item["editing_type_id"] not in edit_category_list:
        continue

    original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
    editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
    image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
    editing_instruction = item["editing_instruction"]
    blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []

    genre=random.randint(0,9)
    artist=random.randint(0,14)
    style=random.randint(0,18)

    genre_dict[genre_list[genre]] += 1
    artist_dict[artist_list[artist]] += 1
    style_dict[style_list[style]] += 1

    data={
        "image_path":item['image_path'],
        "source_prompt":original_prompt,
        "genre_class":genre,
        "artist_class":artist,
        "style_class":style
    }

    img_prompt_dict[key]=data

print("genre_dict:",genre_dict)
print("artist_dict:",artist_dict)
print("style_dict:",style_dict)

image700_source2edit_prompt=json.dumps(img_prompt_dict,indent=4)
output_path1=os.path.join(output_path,"image700_source2edit_prompt.json")
if not os.path.exists(os.path.dirname(output_path1)):
        os.makedirs(os.path.dirname(output_path1))
with open(output_path1,'w') as file:
    file.write(image700_source2edit_prompt)

genre_count=json.dumps(genre_dict,indent=4)
output_path2=os.path.join(output_path,"genre_count.json")
with open(output_path2,'w') as file:
    file.write(genre_count)

artist_count=json.dumps(artist_dict,indent=4)
output_path3=os.path.join(output_path,"artist_count.json")
with open(output_path3,'w') as file:
    file.write(artist_count)

style_count=json.dumps(style_dict,indent=4)
output_path4=os.path.join(output_path,"style_count.json")
with open(output_path4,'w') as file:
    file.write(style_count)