'''
@File    :   dataset.py
@Time    :   2021/08/10 15:38:07
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import torch
import cv2
import glob
import os
from torchvision import transforms
import torch.utils.data as data
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import random


class CNV(data.Dataset):
    def __init__(self, dataset_path, scale=(512,512), k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path = dataset_path+'\\img'
        self.images_list, self.labels_list = self.read_list(k_fold_test=k_fold_test)
        self.seq = iaa.Sequential([
            iaa.Fliplr(0.5),                                                 # 水平翻转
            iaa.SomeOf(n=(0,2),children=[                                   
                iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},                # 尺度缩放
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移
                    rotate=(-10, 10)),                                       # 旋转
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),
                    iaa.AverageBlur(k=(0, 3)),
                    iaa.MedianBlur(k=(1, 3))]),
                iaa.AdditiveGaussianNoise(scale=(0.0, 0.06 * 255)),  # 高斯噪声
                iaa.contrast.LinearContrast((0.9, 1.1))
            ],random_order=True)
        ])
        self.resize_label = transforms.Resize(scale, Image.NEAREST)   # 标签缩放（最近邻插值）[对于标签不需要过高质量]
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)    # 图像缩放（双线性插值）
        self.to_tensor = transforms.ToTensor()                          # Image对象转Tensor


    def __getitem__(self, idx):
        # load image
        img = Image.open(self.images_list[idx])
        img = self.resize_img(img)
        img = np.array(img)
        # load label
        label = Image.open(self.labels_list[idx])
        label = self.resize_label(label)
        label = np.array(label)
        # label = np.ones(shape=(label.shape[0],label.shape[1]),dtype=np.uint8)
        label[label != 255] = 0
        label[label == 255] = 1
        # augment image and label
        if(self.mode == 'train'):  # 训练时对图像和标签数据增强
            seq_det = self.seq.to_deterministic()  # 创建数据增强的序列
            segmap = ia.SegmentationMapsOnImage(label, shape=label.shape) # 将分割结果转换为SegmentationMapOnImage类型，方便后面可视化
            img = seq_det.augment_image(img) # 对图像进行数据增强
            label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8) # 将数据增强应用在分割标签上，并且转换成np类型
            label = np.reshape(label, (1,)+label.shape)
            label = torch.from_numpy(label.copy()).float()

        elif self.mode=='val':
            label = np.reshape(label, (1,)+label.shape)
            label = torch.from_numpy(label.copy()).float()
            label = label,self.labels_list[idx]

        img = self.to_tensor(img.copy()).float()

        return img, label

    def __len__(self):
        return len(self.images_list)

    def read_list(self, k_fold_test=1):
        fold = sorted(os.listdir(self.img_path))
        img_list = []
        label_list = []
        if self.mode == 'train':
            fold_r = fold
            fold_r.remove('f' + str(k_fold_test))  # remove testdata
            for item in fold_r:
                for one_pat in os.listdir(os.path.join(self.img_path,item)):
                    for one_img in os.listdir(os.path.join(self.img_path,item,one_pat)):
                        img_list += os.path.join(self.img_path,item,one_pat,one_img)
            label_list = [x.replace('img', 'mask') for x in img_list]

        elif self.mode == 'val':
            fold_s = fold[k_fold_test - 1]
            for one_pat in os.listdir(os.path.join(self.img_path,fold_s)):
                for one_img in os.listdir(os.path.join(self.img_path,item,one_pat)):
                    img_list += os.path.join(self.img_path,item,one_pat,one_img)
            label_list = [x.replace('img', 'mask') for x in img_list]
        return img_list, label_list


if __name__ == '__main__':
    dataset_path = r"F:\Dataset\ZSQ\ZSQ_seg_png_4_fold"
    cnv_dataset = CNV(dataset_path=dataset_path,k_fold_test=1, mode='train')
    img_list,label_list = cnv_dataset.read_list(1)
    # print(img_list)
    dataloader = data.DataLoader(cnv_dataset,1)
    for index,(data,label) in enumerate(dataloader):
        print(data.shape)
        print(label.shape)