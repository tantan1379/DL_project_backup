import sys
sys.path.append("../")
sys.path.append("../..")
import torch
from torch import nn
from torch.nn import functional as F
from models.nets import extractors as extractors
import sys
# import extractors

'''
# Network structure can be seen in '../network_structure_img/PSPNet.jpg'
PSPNet为像素级场景解析提供了有效的全局上下文先验
金字塔池化模块可以收集具有层级的信息，比全局池化更有代表性
在计算量方面，我们的PSPNet并没有比原来的空洞卷积FCN网络有很大的增加
在端到端学习中，全局金字塔池化模块和局部FCN特征可以被同时训练
'''

# 金字塔池化模块
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]   # list+[] 相当于append
        bottle = self.bottleneck(torch.cat(priors, 1))  # torch.cat的第一个参数需要是一个list或tuple，而nn.ModuleList()返回的就是一个list
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)
        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)
        p = p + sc
        p2 = self.conv2(p)
        return p + p2


class Backbone(nn.Module):
    def __init__(self, num_classes=1,in_channels=1, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.feats = getattr(extractors,backend)(pretrained=pretrained)# getattr获取py文件中的函数地址，extractor为resnet34的特征提取层
        self.psp = PSPModule(psp_size, 1024, sizes)

        self.up_1 = PSPUpsample(1024, 1024+64, 512)
        self.up_2 = PSPUpsample(512, 512+64, 256)
        self.up_3 = PSPUpsample(256, 256+1, 64)


        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x1 = self.conv1(x)          # 64,256,256
        f = self.feats.bn1(x1)
        f = self.feats.relu(f)
        f = self.feats.maxpool(f)   # 64, 128, 128
        x2 = self.feats.layer1(f)    # 64, 128, 128
        f = self.feats.layer2(x2)    # 128, 64, 64
        f = self.feats.layer3(f)    # 256, 64, 64
        f = self.feats.layer4(f)    # 512, 64, 64
        p = self.psp(f)             # 1024, 64, 64

        p = self.up_1(p,x2)         # 512,128,128

        p = self.up_2(p,x1)         # 256,256,256

        p = self.up_3(p,x)          # 32,512,512

        p = self.final(p)
        return p


if __name__ == "__main__":
    input_img = torch.rand((4,1,256,256)).cuda()
    net = Backbone(num_classes=1,in_channels=1).cuda()
    out = net(input_img)