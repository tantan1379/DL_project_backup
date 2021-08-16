import sys
sys.path.append("..")
sys.path.append("../..")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from models.utils.layers import conv1x1,BasicBlock,Bottleneck,resnet_unetUp
from models.utils.init_weights import init_weights
from models.nets.resnet import resnet18,resnet34,resnet50
#from .utils import load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



class DecoderBlock(nn.Module):
    def __init__(self,in_channels,n_filters):
        super(DecoderBlock,self).__init__()
        self.conv1 = nn.Sequential(   
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels//4,n_filters,1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x



class ResUNet_concat(nn.Module):

    def __init__(self, in_channels=1, num_classes=1, is_deconv=True,
                 pretrained=True,norm_layer=None):
        super(ResUNet_concat, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.backbone = resnet34(pretrained=pretrained)
        
        #downsampling
        if in_channels==3:
            self.conv1 = self.backbone.conv1
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        #upsampling
        #Deconv上采样时通道数减半，in_channels=[512,256,128,96,16]
        #upsample上采样时通道数不变，in_channels=[512+256,256+128,128+64,96+64,32+0]
        self.up_concat3 = resnet_unetUp(in_size=512, out_size=256, skip_size=256, is_deconv=self.is_deconv)
        # self.up_concat3 = resnet_unetUp(in_size=2048, out_size=1024, skip_size=1024, is_deconv=self.is_deconv)
        self.up_concat2 = resnet_unetUp(in_size=256, out_size=128, skip_size=128, is_deconv=self.is_deconv)
        # self.up_concat2 = resnet_unetUp(in_size=1024, out_size=512, skip_size=512, is_deconv=self.is_deconv)
        self.up_concat1 = resnet_unetUp(in_size=128, out_size=64, skip_size=64, is_deconv=self.is_deconv)
        # self.up_concat1 = resnet_unetUp(in_size=512, out_size=256, skip_size=256, is_deconv=self.is_deconv)
        self.up_concatcbr = resnet_unetUp(in_size=64, out_size=32, skip_size=64, is_deconv=self.is_deconv)
        # self.up_concatcbr = resnet_unetUp(in_size=256, out_size=128, skip_size=64, is_deconv=self.is_deconv)
        self.up_sample = resnet_unetUp(in_size=32, out_size=16, skip_size=0, is_deconv=self.is_deconv)  # 没有skip，仅上采样至原图大小
        # self.up_sample = resnet_unetUp(in_size=128, out_size=64, skip_size=0, is_deconv=self.is_deconv) #没有skip，仅上采样至原图大小
        #segmentation_head
        self.final = nn.Conv2d(16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.final = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

        # initialise weights 这里mode='fan_in'or'fan_out'可以改
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            

    def forward(self, x):                # 原图:3*512*256
        x = self.conv1(x)                # 64*256*128（步长为2的7*7卷积）
        x = self.bn1(x)
        c1 = self.relu(x)                    
        x = self.maxpool(c1)             # 64*128*64
        c2 = self.layer1(x)              # 64*128*64
        c3 = self.layer2(c2)             # 128*64*32
        c4 = self.layer3(c3)             # 256*32*16
        c5 = self.layer4(c4)             # 512*16*8
        up4 = self.up_concat3(c5,c4)     # 256*32*16
        up3 = self.up_concat2(up4,c3)    # 128*64*32
        up2 = self.up_concat1(up3,c2)    # 64*128*64
        up1 = self.up_concatcbr(up2,c1)  # 32*256*128
        up0 = self.up_sample(up1,None)   # 16*512*256（没有skip,仅上采样）
        final = self.final(up0)          # 1*512*256
        final = torch.sigmoid(final)
        return final


class ResUNet_add(nn.Module):
    def __init__(self,in_channels=3,num_classes=1):
        super(ResUNet_add,self).__init__()

        filters = [64,128,256,512]
        resnet = resnet34(pretrained=True)
        if in_channels==3:
            self.firstconv = resnet.conv1  # h/2,w/2
        else:
            self.firstconv = nn.Conv2d(in_channels,64,7,2,3,bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # h/2,w/2

        self.encoder1 = resnet.layer1 # h、w、c都不变
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512,filters[2])
        self.decoder3 = DecoderBlock(filters[2],filters[1])
        self.decoder2 = DecoderBlock(filters[1],filters[0])
        self.decoder1 = DecoderBlock(filters[0],filters[0])
        
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU()
        self.finalconv2 = nn.Conv2d(32, 32, 3, stride=1,padding=1)
        self.finalrelu2 = nn.ReLU()
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, stride=1, padding=1)
        # self.final = nn.Conv2d(filters[0],num_classes,1)


    def forward(self,x):  # 3x512x256
        # encoder
        x1 = self.firstconv(x)  # 64x256x128
        x2 = self.firstbn(x1)
        e0 = self.firstrelu(x2)

        x4 = self.firstmaxpool(e0)  # 64x128x64
        e1 = self.encoder1(x4)  # 64x128x64
        e2 = self.encoder2(e1)  # 128x64x32
        e3 = self.encoder3(e2)  # 256x32x16
        e4 = self.encoder4(e3)  # 512x16x8
        # decoder
        d4 = self.decoder4(e4) + e3  # 256x32x16 + 256x32x16
        d3 = self.decoder3(d4) + e2  # 128x64x32 + 128x64x32
        d2 = self.decoder2(d3) + e1  # 64x128x64 + 64x128x64\
        d1 = self.decoder1(d2) # 64x256x128 + 64x256x128
        out = self.finaldeconv1(d1)  # 32x512x256
        out = self.finalrelu1(out)
        out = self.finalconv2(out)   # 32x512x256
        out = self.finalrelu2(out)
        out = self.finalconv3(out)   # 1x512x256
        out = torch.sigmoid(out)
        return out




#------------------------------------------------------------------------------------------------------
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = ResUNet_concat(in_channels=3, num_classes=1, is_deconv=True, pretrained=True).cuda()
    x = torch.rand((4, 3, 512, 256)).cuda()
    y = net(x)
    # print(y.shape)