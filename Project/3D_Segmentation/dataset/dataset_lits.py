from torch.utils.data import Dataset, DataLoader
import os
import torch
from torchvision import transforms as T
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from utils.common import *
import sys
sys.path.append("..")


class Lits_DataSet(Dataset):
    def __init__(self, crop_size, resize_scale, dataset_path, mode=None):
        self.crop_size = crop_size
        self.resize_scale = resize_scale
        self.dataset_path = dataset_path
        self.mode = mode

        if self.mode == 'train':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'train_name_list.txt'))
        elif self.mode == 'val':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'val_name_list.txt'))
        elif self.mode == 'test':
            self.filename_list = load_file_name_list(
                os.path.join(dataset_path, 'test_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

    def __getitem__(self, index):
        data, target=self.get_item_by_index(crop_size = self.crop_size, index = index,
                                              resize_scale = self.resize_scale)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.filename_list)

    def get_item_by_index(self, crop_size, index, resize_scale = 1):
        img, label=self.get_np_data_3d(
            self.filename_list[index], resize_scale = resize_scale)
        if self.mode == "train":
            # img, label = center_crop_3d(img, label, slice_num=32)
            img, label=random_crop_3d(img, label, crop_size)
        if self.mode == "val" or self.mode == "test":
            img, label=center_crop_3d(img, label, slice_num = 32)  # 取16个lices
        return np.expand_dims(img, axis = 0), label

    def get_np_data_3d(self, filename, resize_scale = 1):
        data_np=sitk.GetArrayFromImage(sitk.ReadImage(
            self.dataset_path + '/data/' + filename, sitk.sitkInt16))
        if self.resize_scale != 1.0:
            data_np=ndimage.zoom(
                data_np, zoom = self.resize_scale, order = 3)  # 双三次重采样
        data_np=data_np.astype(np.float32)
        data_np=data_np/1000.0
        label_np=sitk.GetArrayFromImage(sitk.ReadImage(
            self.dataset_path + '/label/' + filename.replace('volume', 'segmentation'), sitk.sitkInt8))
        if self.resize_scale != 1.0:
            label_np=ndimage.zoom(
                label_np, zoom = self.resize_scale, order = 0)  # 最近邻重采样
        # print(data_np.shape,label_np.shape)
        return data_np, label_np


# 测试代码
if __name__ == '__main__':
    fixd_path='../fixed'
    dataset=Lits_DataSet([16, 64, 64], 0.5, fixd_path,
                           mode = 'train')  # batch size
    data_loader=DataLoader(
        dataset = dataset, batch_size = 2, num_workers = 1, shuffle = True)
    for batch_idx, (data, target) in enumerate(data_loader):
        target=to_one_hot_3d(target.long())
        # print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0, 0, 0])
        plt.subplot(122)
        plt.imshow(target[0, 1, 0])
        plt.show()
