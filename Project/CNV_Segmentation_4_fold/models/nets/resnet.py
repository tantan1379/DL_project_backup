'''
@File    :   resnet.py
@Time    :   2021/07/05 14:56:49
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.
'''

'''
notes:
    resnet的主要结构都可以分为4个layer，每个layer都对通道数进行调整，18/34为64->128->256->512,50/101/152为64->256->512->1024->2048;
    注意bottleneck结构的每个layer的第一个block会对通道进行调整(×2)，为inplanes->inplanes/2->inplanes*2，同时对特征图的大小进行缩减，因此shortcut也要做对应的下采样；
    除了第一个block的其余block通道数输入输出前后不变，为inplanes->inplanes/4->inplanes
'''

import sys
sys.path.append("..")
sys.path.append("../..")
import torch.nn as nn
import torch
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
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1 # 表示bottleneck的维度变化倍数，basicblock中两层conv的维度不变化

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
        # 对每个layer的第一个block的第一个输入维度进行调整操作后再与输出相加
        if self.downsample is not None:
            residual = self.downsample(x)
        # 其他block则直接相加
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 # 表示bottleneck的维度变化倍数，bottleneck第3个conv比第二个conv的通道数扩大4倍
    # inplanes是每个block的输入维度，planes为输出维度
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # 先缩小
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False) # 第三个conv比第二个conv扩大expansion倍
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
        # 对每个layer的第一个block的第一个输入维度进行调整操作后再与输出相加
        if self.downsample is not None:
            residual = self.downsample(x)
        # 其他block则直接相加
        out += residual
        out = self.relu(out)

        return out

'''
注意点（针对bottleneck结构的resnet)：
每个layer均由若干个block组成，每个layer的第一个block需要对特征图进行维度的调整，其余的block仅仅是对维度的先缩再放）
除了第一个layer之外，每个layer的第一个block还需要对特征图进行下采样，因此同时需要对shortcut进行下采样以匹配block输出的特征图
因此我们需要对第一个block进行单独处理
'''

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
        self.layer1 = self._make_layer(block, 64, layers[0],stride=1)
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

    # planes为每一个layer的基础通道数，blocks_num为每一个layer的block数，stride为shortcut的步距（表示是否下采样）
    def _make_layer(self, block, planes, blocks_num, stride=1):
        downsample = None
        # stride!=1 表示shortcut的步距不为1，也就是指layer234，这三层需要对shortcut进行下采样；
        # self.inplanes!=planes*block.expansion表示该block的输入通道数和输出通道数不同，也就是除第一个layer外每个layer的第一个block
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), # 将shortcut的输入通道数转换为输出通道数，同时对layer234的shotrcut进行下采样
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # layer的第一个block单独处理（包括stride和shortcut的downsample）
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 其余block的输入变为原先基础通道的4倍
        self.inplanes = planes * block.expansion
        # layer的其余部分为block的重复(inplanes->inplanes/4->inplanes)*blocks_num，重复次数设为blocks_num-1（basicblock通道不变，bottleneck通道先减少再增加
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x) # 64x256x128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 64x128x64
        x = self.layer1(x)  # 64x128x64
        x = self.layer2(x)  # 128x64x32
        x = self.layer3(x)  # 256x32x16
        t = self.layer4(x)  # 512x16x8
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
        pretrained_dict = torch.load('../../Pretrained_model/resnet18.pth')
        model.load_state_dict(pretrained_dict)
        print('Petrained Model Have been loaded!')
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()


    if pretrained:
        pretrained_dict = torch.load('../Pretrained_model/resnet34.pth')
        model.load_state_dict(pretrained_dict)
        print('Petrained Model Have been loaded!')
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = torch.load('../Pretrained_model/resnet50.pth')
        model.load_state_dict(pretrained_dict)
        print('Petrained Model Have been loaded!')
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


if __name__ == '__main__':
    model = resnet18(pretrained=True).cuda()
    input_img = torch.Tensor(1,3,512,256).cuda()
    output_img = model(input_img)
    # print(output_img.shape)