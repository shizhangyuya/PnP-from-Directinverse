import torch
import argparse
import os
from models import *

a=torch.cat([torch.zeros(4,20,20),torch.ones(4,20,20),2*torch.ones(4,20,20)],dim=0)
b=a[2*[1,2],:]
# print(a)
# print(b)
# print(a.shape)
# print(b.shape)
# print([1]*3+[2]*3)
# clip_length=4
# index_list = []
# # frame_index = []
# for frame in [0, 1]:
#     for i in range(3):
#         frame_index=[]
#         frame_index+=[frame+i*clip_length] * clip_length
#         index_list.append(frame_index)
# c=torch.cat([a[frame_index,:] for frame_index in index_list],dim=1)
# print(c.shape)

# key = torch.cat([   key[:, frame_index] for frame_index in frame_index_list
# print(list(range(3)))


# batch_size=24
# clip_length=int(batch_size/3)
# self_list=list(range(batch_size))
# prev_list=[0]+self_list[:clip_length-1]+[clip_length]+self_list[clip_length:clip_length*2-1]+[clip_length*2]+self_list[clip_length*2:clip_length*3-1]
# next_list=self_list[1:clip_length]+[clip_length-1]+self_list[clip_length+1:clip_length*2]+[clip_length*2-1]+self_list[clip_length*2+1:clip_length*3]+[clip_length*3-1]
# print(len(self_list),self_list)
# print(len(prev_list),prev_list)
# print(len(next_list),next_list)
# main=3

# def func3():
#     print(m)

# if __name__ =="__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--test', action="store_false")  
#     parser.add_argument('--list',nargs='+',type=str,default=['prev','next'])
#     args = parser.parse_args()


#     te=args.test
#     list=args.list
#     print(list)


print(os.getcwd())