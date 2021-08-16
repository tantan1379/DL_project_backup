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

class CE_Backbone(nn.Module):
    
    def __init__(self, in_channels=3,num_classes=1):
        super(CE_Backbone, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        if in_channels==3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.spp = SPPblock(512,512)

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


    def forward(self, x): # 3x512x256
        # Encoder
        x = self.firstconv(x)        # 64x256x128
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)     # 64x128x64
        e1 = self.encoder1(x)        # 64x128x64
        e2 = self.encoder2(e1)       # 128x64x32
        e3 = self.encoder3(e2)       # 256x32x16
        e4 = self.encoder4(e3)       # 512x16x8
        # Center
        e4 = self.spp(e4)            # 512x16x8
        # Decoder
        d4 = self.decoder4(e4) + e3  # 256x32x16
        d3 = self.decoder3(d4) + e2  # 128x64x32
        d2 = self.decoder2(d3) + e1  # 64x128x64
        d1 = self.decoder1(d2)       # 32x256x128

        out = self.finaldeconv1(d1)  # 32x512x256
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # 32x512x256
        out = self.finalrelu2(out)
        final = self.finalconv3(out)   # 1x512x256
        final = torch.sigmoid(final)
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
    net = CE_Backbone(in_channels=3,n_classes=1).cuda()
    output = net(x)
    # forward = net.forward(x)
    # print(forward)
    # print(type(forward))