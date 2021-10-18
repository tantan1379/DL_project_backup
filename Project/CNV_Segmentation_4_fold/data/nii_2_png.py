'''
@File    :   make_dataset.py
@Time    :   2021/08/10 11:08:20
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import os
import glob
import shutil
import cv2
import numpy as np
import SimpleITK as sitk
# import matplotlib.pyplot as plt

def double_linear(input_signal, zoom_multiples):
    input_signal_cp = np.copy(input_signal)
    input_row, input_col = input_signal_cp.shape
    output_row = int(input_row*zoom_multiples)
    output_col = int(input_col*zoom_multiples)

    output_signal = np.zeros((output_row, output_col))
    for i in range(output_row):
        for j in range(output_col):
            temp_x = i/output_row*input_row
            temp_y = j/output_col*input_col
            x1 = int(temp_x)
            y1 = int(temp_y)
            x2 = x1
            y2 = y1+1
            x3 = x1+1
            y3 = y1
            x4 = x1+1
            y4 = y1+1
            u = temp_x-x1
            v = temp_y-y1
            if x4 >= input_row:
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col:
               y4 = input_col - 1
               y3 = y4
               y1 = y4 - 1
               y2 = y4 - 1
            # 插值
            output_signal[i, j] = (1-u)*(1-v)*int(input_signal_cp[x1, y1]) + (1-u)*v*int(input_signal_cp[x2, y2]) + u*(1-v)*int(input_signal_cp[x3, y3]) + u*v*int(input_signal_cp[x4, y4])
    return output_signal




select_path = "F:\Dataset\ZSQ\ZSQ_selected"
des_path = "F:/Dataset/ZSQ/ZSQ_seg_png"
origin_path = "F:/Dataset/ZSQ/ZSQ_seg_origin/img"
origin_mask_path = "F:/Dataset/ZSQ/ZSQ_seg_origin/mask"

# make files
pat_list = list()
for one_pat in os.listdir(origin_path):
    pat_list.append(one_pat)


for one_pat in os.listdir(origin_path):
    for i in range(4):
        if not os.path.exists(os.path.join(des_path,"img",one_pat,one_pat+"_time_"+str(i+1))):
            os.makedirs(os.path.join(des_path,"img",one_pat,one_pat+"_time_"+str(i+1)))

for one_pat in os.listdir(origin_path):
    for i in range(4):
        if not os.path.exists(os.path.join(des_path,"mask",one_pat,one_pat+"_time_"+str(i+1))):
            os.makedirs(os.path.join(des_path,"mask",one_pat,one_pat+"_time_"+str(i+1)))

# # rename
# for one_pat in os.listdir(select_path):
#     for i,one_time in enumerate(os.listdir(os.path.join(select_path,one_pat))):
#         os.rename(os.path.join(select_path,one_pat,one_time),os.path.join(select_path,one_pat,one_pat+"_time_"+str(i+1)))


# for one_pat in os.listdir(select_path):
#     for i,one_time in enumerate(os.listdir(os.path.join(select_path,one_pat))):
#         for file in os.listdir(os.path.join(select_path,one_pat,one_time)):
#             if 'nii.gz' not in file:
#                 name = file
#                 os.rename(os.path.join(select_path,one_pat,one_time,file),os.path.join(select_path,one_pat,one_time,file+'.nii.gz'))

# remove
# for one_pat in os.listdir(origin_path):
#     for i,one_time in enumerate(os.listdir(os.path.join(origin_path,one_pat))):
#         for file in os.listdir(os.path.join(origin_path,one_pat,one_time)):
#             if 'seg' in file:
#                 os.remove(os.path.join(origin_path,one_pat,one_time,file))

# for one_pat in os.listdir(origin_mask_path):
#     for i,one_time in enumerate(os.listdir(os.path.join(origin_mask_path,one_pat))):
#         for file in sorted(os.listdir(os.path.join(origin_mask_path,one_pat,one_time))):
#             if 'seg' not in file:
#                 name = file
#                 os.remove(os.path.join(origin_mask_path,one_pat,one_time,file))
#             else:
#                 os.rename(os.path.join(origin_mask_path,one_pat,one_time,file),os.path.join(origin_mask_path,one_pat,one_time,name))


# # # convert img to png
# for idx,one_pat in sorted(enumerate(os.listdir(origin_path))):
#     if idx==0:
#         for i,one_time in enumerate(os.listdir(os.path.join(origin_path,one_pat))):
#             for file in os.listdir(os.path.join(origin_path,one_pat,one_time)):
#                 img_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(origin_path,one_pat,one_time,file)))
#                 for slice_index,one_slice_array in enumerate(img_array):
#                     png_name = one_pat+"_time_"+str(i+1)+"_"+str(slice_index+1)
#                     des_file_path = os.path.join(des_path,"img",one_pat,one_pat+"_time_"+str(i+1),png_name+'.png')
#                     cv2.imwrite(des_file_path,one_slice_array)

# convert label to png
for idx,one_pat in sorted(enumerate(os.listdir(origin_mask_path))):
    if idx==0:
        for i,one_time in enumerate(os.listdir(os.path.join(origin_mask_path,one_pat))):
            for file in os.listdir(os.path.join(origin_mask_path,one_pat,one_time)):
                img_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(origin_mask_path,one_pat,one_time,file)))
                print(os.path.join(origin_mask_path,one_pat,one_time,file))
                for slice_index,one_slice_array in enumerate(img_array):
                    one_slice_array = double_linear(one_slice_array,1)
                    one_slice_array[one_slice_array!=1]=0.0
                    one_slice_array[one_slice_array==1]=255.0
                    png_name = one_pat+"_time_"+str(i+1)+"_"+str(slice_index+1)
                    des_file_path = os.path.join(des_path,"mask",one_pat,one_pat+"_time_"+str(i+1),png_name+'.png')
                    cv2.imwrite(des_file_path,one_slice_array)