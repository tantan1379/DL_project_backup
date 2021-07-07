import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage


class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, classes):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = classes  # 分割类别数（只分割肝脏，或者分割肝脏和肿瘤）
        self.upper = 1000
        self.lower = 0
        self.expand_slice = 20  # 轴向外侧扩张的slice数量
        self.size = 48  # 取样的slice数量
        self.x_down_scale = 0.5
        self.y_down_scale = 0.5
        self.slice_thickness = 1

    def fix_data(self):
        if not os.path.exists(self.fixed_path):    # 创建保存目录
            os.makedirs(self.fixed_path+'data')
            os.makedirs(self.fixed_path+'label')
        file_list = os.listdir(self.raw_root_path + 'data/')
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)
        for ct_file, i in zip(file_list, range(Numbers)):
            print(ct_file, " | {}/{}".format(i+1, Numbers))
            ct_path = os.path.join(self.raw_root_path + 'data/', ct_file)
            seg_path = os.path.join(
                self.raw_root_path + 'label/', ct_file.replace('volume', 'segmentation'))
            new_ct, new_seg = self.process(
                ct_path, seg_path, classes=self.classes)
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(
                    self.fixed_path + 'data/', ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path + 'label/',
                                ct_file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))
        self.ct_file = ct_file

    def resize_image_itk(self, itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        newSize = np.array(newSize, float)
        factor = originSize / newSize
        newSpacing = originSpacing * factor
        newSize = newSize.astype(np.int)  # spacing肯定不能是整数
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(newSize.tolist())
        resampler.SetOutputSpacing(newSpacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        return itkimgResampled

    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        _, _, slicenum = ct.GetSize()
        ct = self.resize_image_itk(
            ct, (256, 256, slicenum), resamplemethod=sitk.sitkNearestNeighbor)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg = self.resize_image_itk(
            seg, (256, 256, slicenum), resamplemethod=sitk.sitkNearestNeighbor)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:", ct_array.shape, seg_array.shape)
        if classes == 2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1
        # 将灰度值在阈值之外的截断掉
        # ct_array[ct_array > self.upper] = self.upper
        # ct_array[ct_array < self.lower] = self.lower

        # 降采样，（只对x和y轴进行降采样，slice轴的spacing进行归一化）
        ct_array = ndimage.zoom(
            ct_array, (self.slice_thickness, self.y_down_scale, self.x_down_scale), order=3)
        seg_array = ndimage.zoom(seg_array, (self.slice_thickness, self.y_down_scale, self.x_down_scale), order=0)

        # 找到肝脏区域开始和结束的slice，并各向外扩张
        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张个slice
        if start_slice - self.expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= self.expand_slice

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1
        else:
            end_slice += self.expand_slice

        print("Cut out range:", str(start_slice) + '--' + str(end_slice))
        # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
        if end_slice - start_slice + 1 < self.size:
            print(self.ct_file, 'too little slice，give up the sample')
            return None, None
        # 截取保留区域
        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:", ct_array.shape, seg_array.shape)
        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.x_down_scale), ct.GetSpacing()[
                          1] * int(1 / self.y_down_scale), self.slice_thickness))

        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.x_down_scale), ct.GetSpacing()[
                           1] * int(1 / self.y_down_scale), self.slice_thickness))
        return new_ct, new_seg

    def write_train_val_test_name_list(self):
        data_name_list = os.listdir(self.fixed_path + "/" + "data")
        test_list = os.listdir("F:\\Dataset\\gliomas\\batch1\\data")
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)
        test_num = len(test_list)
        print('test dataset total numbers of samples is :', test_num)

        train_rate = 0.75
        val_rate = 0.25

        assert val_rate+train_rate == 1.0
        train_name_list = data_name_list[0:int(data_num*train_rate)]
        val_name_list = data_name_list[int(
            data_num*train_rate):int(data_num*(train_rate + val_rate))]
        test_name_list = test_list[:]

        self.write_name_list(train_name_list, "train_name_list.txt")
        self.write_name_list(val_name_list, "val_name_list.txt")
        self.write_name_list(test_name_list, "test_name_list.txt")

    def write_name_list(self, name_list, file_name):
        f = open(self.fixed_path + file_name, 'w')
        for i in range(len(name_list)):
            f.write(str(name_list[i]) + "\n")
        f.close()


if __name__ == '__main__':
    raw_dataset_path = 'F:\\Dataset\\gliomas\\batch1\\'
    fixed_dataset_path = './fixed/'
    classes = 2  # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, classes)
    tool.fix_data()                            # 对原始图像进行修剪并保存
    tool.write_train_val_test_name_list()      # 创建索引txt文件
