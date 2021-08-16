

# **神经网络笔记**

## Unet

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210603092101649.png" alt="image-20210603092101649" style="zoom:50%;" />

1、优势

* 利用FCN的结构作为编码器，通过逐级的卷积层和下采样层提取图像中的高维语义特征；

* 使用一种自底向上的解码器逐阶段恢复由编码器产生的空间信息；
* 使用跳跃连接弥补在下采样过程中产生的信息丢失。

2、缺陷（cpfnet分析）

* 当由编码器深层产生的上下文信息传递到浅层时，由于每个单阶段的特征提取能力较弱，特征容易被稀释；

* 简单的跳跃连接忽视了每个阶段的全局信息，而且这是一种针对局部信息的无差别融合，将会引入不相关的信息影响分割的结果；
* 在每个单独的阶段，缺少对多尺度上下文信息的有效提取。



```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=False, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x/self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels,filters[0],self.is_batchnorm)
        self.conv2 = unetConv2(filters[0],filters[1],self.is_batchnorm)
        self.conv3 = unetConv2(filters[1],filters[2],self.is_batchnorm)
        self.conv4 = unetConv2(filters[2],filters[3],self.is_batchnorm)
        self.center = unetConv2(filters[3],filters[4],self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4],filters[3],self.is_deconv)
        self.up_concat3 = unetUp(filters[3],filters[2],self.is_deconv)
        self.up_concat2 = unetUp(filters[2],filters[1],self.is_deconv)
        self.up_concat1 = unetUp(filters[1],filters[0],self.is_deconv)
        # final conv
        self.final = nn.Conv2d(filters[0],n_classes,1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init_weights(m,'kaiming')
            elif isinstance(m,nn.BatchNorm2d):
                init_weights(m,'kaiming')
        
    def forward(self,inputs):               # 3*256*512                          
        # encoder                           
        conv1 = self.conv1(inputs)          # 32*256*512
        maxpool1 = self.maxpool(conv1)      # 32*128*256
        conv2 = self.conv2(maxpool1)        # 64*128*256
        maxpool2 = self.maxpool(conv2)      # 64*64*128
        conv3 = self.conv3(maxpool2)        # 128*64*128
        maxpool3 = self.maxpool(conv3)      # 128*32*64
        conv4 = self.conv4(maxpool3)        # 256*32*64
        maxpool4 = self.maxpool(conv4)      # 256*16*32
        center = self.center(maxpool4)      # 512*16*32
        # decoder(转置卷积+拼接+卷积块)
        up4 = self.up_concat4(center,conv4) # 512*16*32(ConvTranspose)->256*32*64(concat)->512*32*64(conv2)->256*32*64
        up3 = self.up_concat3(up4,conv3)    # 512*16*32(ConvTranspose)->128*64*128(concat)->256*64*128(conv2)->128*64*128
        up2 = self.up_concat2(up3,conv2)    # 512*16*32(ConvTranspose)->64*128*256(concat)->128*128*256(conv2)->64*128*256
        up1 = self.up_concat1(up2,conv1)    # 512*16*32(ConvTranspose)->32*256*512(concat)->64*256*512(conv2)->32*256*512
        final = self.final(up1)             # 1*256*512
        return final                        # 1*256*512

class unetConv2(nn.Module):
    def __init__(self,in_size,out_size,is_batchnorm,n=2,ks=3,stride=1,padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks # kernel_size
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1,n+1):
                conv = nn.Sequential(nn.Conv2d(in_size,out_size,ks,s,p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self,'conv%d'%i,conv)
                in_size = out_size
        
        else:
            for i in range(1,n+1):
                conv = nn.Sequential(nn.Conv2d(in_size,out_size,ks,s,p),
                                     nn.ReLU(inplace=True))
                setattr(self,'conv%d'%i,conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m,init_type="kaiming")
    
    def forward(self,inputs):
        x = inputs
        for i in range(1,self.n+1):
            conv = getattr(self,'conv%d'%i)
            x = conv(x)
        return x

class unetUp(nn.Module):
    def __init__(self,in_size,out_size,is_deconv,n_concat=2):
        super(unetUp,self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size,out_size,False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size,out_size,kernel_size=2,stride=2,padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size,out_size,1)
            )
        
        # initialize the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2')!=-1:
                continue
            init_weights(m,init_type='kaiming')
        
    def forward(self,high_feature,*low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0,feature],dim=1)  # dim=1是通道维
        return self.conv(outputs0)

# 权重初始化（对于激活函数ReLU一般采用何凯明提出的kaiming正态分布初始化卷积层参数）
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
```





## ResNet



<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210603200737082.png" alt="image-20210603200737082" style="zoom: 67%;" />

```python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchsummary

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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

}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=1000,deep_base=False,stem_width=32):
        self.inplanes = stem_width*2 if deep_base else 64
        
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1= nn.Sequential(
                nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        t = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return t


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/home/wtt/.cache/torch/checkpoints/'))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()


    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['resnet34'], model_dir='/home/wtt/.cache/torch/checkpoints/')
        print('Petrain Model Have been loaded!')
        # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
# net = resnet34(pretrained=False)
# torchsummary.summary(net, (3, 512, 512))
```

网络结构：

```
ResNet(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (4): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (5): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Linear(in_features=512, out_features=1000, bias=True)
  )
)
```

示意图：

<img src="https://img-blog.csdnimg.cn/20190620212741777.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NjMxNTIx,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 25%;" />





<img src="https://img-blog.csdnimg.cn/2019062022131876.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI5NjMxNTIx,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:25%;" />



---

## SE-Net

注意力模块

```python
class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

```



## Attention

Attention机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：

**第一个过程是根据Query和Key计算权重系数**，**第二个过程根据权重系数对Value进行加权求和**。

而第一个过程又可以细分为两个阶段：第一个阶段根据Query和Key计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如图展示的三个阶段。

<img src="C:\Users\TRT\AppData\Roaming\Typora\typora-user-images\image-20210623100552774.png" alt="image-20210623100552774" style="zoom: 33%;" />



在第一个阶段，可以引入不同的函数和计算机制，根据Query和某个 Keyi ，计算两者的相似性或者相关性，最常见的方法包括：求两者的向量点积、求两者的向量Cosine相似性或者通过再引入额外的神经网络来求值。

第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，第二阶段引入类似SoftMax的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为1的概率分布；另一方面也可以通过SoftMax的内在机制更加突出重要元素的权重。

第二阶段的计算结果 ai 即为 Valuei 对应的权重系数，然后进行加权求和即可得到Attention数值。

















