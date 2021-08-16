'''
@File    :   Unet_spp_se.py
@Time    :   2021/07/06 16:37:51
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

import sys
sys.path.append("..")
sys.path.append("../../")
import os
import torch
import torch.nn as nn
from models.nets.resnet import resnet34
from functools import partial
import torch.nn.functional as F
from models.utils.layers import unetConv2,unetUp,unetConv2_dilation,unetUp_cat,unetUp_add
from models.utils.init_weights import init_weights
import math


nonlinearity = partial(F.relu, inplace=True)

class Backbone(nn.Module):
    
    def __init__(self, in_channels=3,n_classes=1,feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(Backbone, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.spp = SPPblock(filters[4],filters[4])

        # upsampling
        self.up_add4 = unetUp_add(filters[4], filters[3], self.is_deconv)
        self.up_add3 = unetUp_add(filters[3], filters[2], self.is_deconv)
        self.up_add2 = unetUp_add(filters[2], filters[1], self.is_deconv)
        self.up_add1 = unetUp_add(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, x): # 3x512x256
        conv1 = self.conv1(x)       # 32x512x256
        maxpool1 = self.maxpool1(conv1)  # 32*256*128

        conv2 = self.conv2(maxpool1)     # 64*256x128
        maxpool2 = self.maxpool2(conv2)  # 64*128*64

        conv3 = self.conv3(maxpool2)     # 128*128*64
        maxpool3 = self.maxpool3(conv3)  # 128*64*32

        conv4 = self.conv4(maxpool3)     # 256*64*32
        maxpool4 = self.maxpool4(conv4)  # 256*32*16

        center = self.center(maxpool4)   # 512*32*16
        spp = self.spp(center)           # 512*32*16
        up4 = self.up_add4(spp,conv4)    # 256*64*32
        up3 = self.up_add3(up4,conv3)    # 128*128*64
        up2 = self.up_add2(up3,conv2)    # 64*256*128
        up1 = self.up_add1(up2,conv1)    # 32*512*256
        final_1 = self.final_1(up1)      # 1*512*256
        final = torch.sigmoid(final_1)
        return final


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 先全局池化，获取全局上下文信息
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), # 通道缩减
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), # 通道扩增
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SPPblock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SPPblock, self).__init__()
        # self.pool1 = nn.MaxPool2d(kernel_size=[1, 1], stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.conv_smoonth = nn.Conv2d(in_channels=3*in_channels+out_channels, out_channels=out_channels, kernel_size=1,bias=False)
        self.se = SELayer(3*in_channels + out_channels)


    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        # self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear',align_corners=True)
        out = torch.cat([self.layer2, self.layer3, self.layer4, x], 1)
        out1 = self.se(out)
        out1 = self.conv_smoonth(out1)
        return out1


if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     net = UNet_1(in_channels=1, n_classes=3, is_deconv=True).cuda()
#     print(net)
    x = torch.rand((1, 3, 512, 256)).cuda()
    net = Backbone(in_channels=3,n_classes=1).cuda()
    output = net(x)
    # forward = net.forward(x)
    # print(forward)
    # print(type(forward))