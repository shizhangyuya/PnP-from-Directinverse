import json
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import random
import os

def add_prefix(word):
    vowels = "aeiouAEIOU"
    if word[0] in vowels:
        return 'an '+ word
    else:
        return 'a ' + word

def prompt_make(source_prompt,genre_class,artist_class,style_class,instruct=False):
    genre_dict = {}
    artist_dict={}
    style_dict={}

    genre_list=[]
    artist_list=[]
    style_list=[]

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

    target_prompt_dict={}
    target_prompt_dict['genre_prompt']=f'{add_prefix(genre_list[genre_class])} of {source_prompt}'.replace('_',' ')
    target_prompt_dict['artist_prompt'] =f'{add_prefix(artist_list[artist_class])} style of {source_prompt}'.replace('_',' ')
    target_prompt_dict['style_prompt'] =f'{add_prefix(style_list[style_class])} style of {source_prompt}'.replace('_',' ')

    instruct_prompt_dict = {}
    instruct_prompt_dict['genre_prompt'] = f'change the image into {genre_list[genre_class]} style'.replace('_',' ')
    instruct_prompt_dict['artist_prompt'] = f'change the image into {artist_list[artist_class]} style'.replace('_',' ')
    instruct_prompt_dict['style_prompt'] = f'change the image into {style_list[style_class]} style'.replace('_',' ')

    if instruct:
        return instruct_prompt_dict
    else:
        return target_prompt_dict