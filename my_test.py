import torch

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
print(list(range(3)))