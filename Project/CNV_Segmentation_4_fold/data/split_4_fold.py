'''
@File    :   split_4_fold.py
@Time    :   2021/08/06 11:00:03
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''


import os
import glob
import shutil

img_path = r"F:\Dataset\ZSQ\ZSQ_seg_png\img"
mask_path = r"F:\Dataset\ZSQ\ZSQ_seg_png\mask"
des_path = r"F:\Dataset\ZSQ\ZSQ_seg_png_4_fold"

pat_list = list()
img_path_list = list()
label_path_list = list()
for one_pat in os.listdir(img_path):
    pat_list.append(one_pat)

# make file
if not os.path.exists(des_path+os.sep+"img"):
    os.mkdir(des_path+os.sep+"img")

if not os.path.exists(des_path+os.sep+"mask"):
    os.mkdir(des_path+os.sep+"mask")

for i in range(1,5):
    if not os.path.exists(des_path+os.sep+"img"+os.sep+"f"+str(i)):
        os.mkdir(des_path+os.sep+"img"+os.sep+"f"+str(i))
    if not os.path.exists(des_path+os.sep+"mask"+os.sep+"f"+str(i)):
        os.mkdir(des_path+os.sep+"mask"+os.sep+"f"+str(i))
