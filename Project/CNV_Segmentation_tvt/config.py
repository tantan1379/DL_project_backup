'''
@File    :   config.py
@Time    :   2021/06/07 16:25:14
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

class cnv_single_config(object):
    num_epochs = 60
    epoch_start_i = 0
    checkpoint_step = 4
    crop_height = 512
    crop_width = 256
    batch_size = 8                  # *
    slice_num = 128
    # 优化器相关
    # optimizer = "SGD"               # Adam/SGD
    lr = 0.01                       # 学习率 *
    lr_mode = 'poly'                # poly优化策略
    img = '2d5'                      # 2d/2d5
    step = 30                       # step模式时的衰减周期
    momentum = 0.9                  # 优化器动量
    weight_decay = 1e-4             # L2正则化系数
    # 训练相关
    net_work = 'pspnet'               # 可选网络 * unet/cpfnet/resunetplusplus/unet_spp_se/pspnet
    mode = 'train_test'             # 训练模式 * train/test/train_test
    num_workers = 0                 # dataloader设置
    num_classes = 1                 # 分割类别数 类别数+加背景 *
    cuda = '0'                      # GPU id选择 *
    pretrained = True
    use_gpu = True
    # 路径相关
    data = 'F:/Dataset/CNV_Seg/'    # 数据存放的根目录 *
    dataset = 'png_split_tvt'      # 数据库名字(需修改成自己的数据名字) *
    log_dirs = './results/save/'          # 存放tensorboard log的文件夹() *
    if img == '2d5':
            save_model_path = '2d5_cnv_seg_using_' + net_work  # 保存模型的文件夹
    else:
        save_model_path = 'cnv_seg_using_' + net_work  # 保存模型的文件夹
    result_path = './results'+'/img_seg/'

class cnv_and_srf_config(object):
    num_epochs = 60
    slice_num = 128
    epoch_start_i = 0
    checkpoint_step = 4
    crop_height = 512
    crop_width = 256
    slice_num = 128
    batch_size = 8                  # *
    # 优化器相关
    # optimizer = "SGD"               # Adam/SGD
    lr = 0.01                       # 学习率 *
    lr_mode = 'poly'                # poly优化策略
    step = 30                       # step模式时的衰减周期
    momentum = 0.9                  # 优化器动量
    weight_decay = 1e-4             # L2正则化系数
    # 训练相关
    net_work = 'unet'               # 可选网络 * unet/cpfnet/resunetplusplus/resunet1
    mode = 'train_test'             # 训练模式 * train/test/train_test
    num_workers = 0                 # dataloader设置
    num_classes = 3                 # 分割类别数 类别数+加背景 *
    img = '2d5'                     # 图像加载方式 2d/2d5
    cuda = '0'                      # GPU id选择 *
    loss = 'SD_Loss'
    use_gpu = True
    pretrained = True
    # 路径相关
    data = 'F:/Dataset/CNV_Seg/'   # 数据存放的根目录 *
    dataset = 'png_split_tvt'      # 数据库名字(需修改成自己的数据名字) *
    log_dirs = './results/save/'          # 存放tensorboard log的文件夹() *
    if img == '2d5':
        save_model_path = '2d5_cnv_and_srf_seg_using_' + net_work  # 保存模型的文件夹
    else:
        save_model_path = 'cnv_and_srf_seg_using_' + net_work  # 保存模型的文件夹
    result_path = './results' + '/img_seg/'