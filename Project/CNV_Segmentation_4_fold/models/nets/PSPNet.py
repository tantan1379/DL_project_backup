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
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size)) # 大小改变，特征维度不变
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)  # 将concat后的特征张量减维
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        p1 = F.interpolate(input=self.stages[0](feats), size=(h, w), mode='bilinear',align_corners=True)
        p2 = F.interpolate(input=self.stages[1](feats), size=(h, w), mode='bilinear',align_corners=True)
        p3 = F.interpolate(input=self.stages[2](feats), size=(h, w), mode='bilinear',align_corners=True)
        p4 = F.interpolate(input=self.stages[3](feats), size=(h, w), mode='bilinear',align_corners=True)
        # priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages] + [feats] # list的‘+’相当于feature的concat
        bottle = self.bottleneck(torch.cat([p1,p2,p3,p4,feats], dim=1))
        out = self.relu(bottle)
        return out


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3) # 每次上采样图像大小扩大一倍
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, num_classes=1,in_channels=1, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.feats = getattr(extractors,backend)(pretrained=pretrained)# getattr获取py文件中的函数地址，extractor为resnet34的特征提取层
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            # nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.conv1(x)
        f = self.feats.bn1(f)
        f = self.feats.relu(f)
        f = self.feats.maxpool(f)   # 64, 128, 64
        f = self.feats.layer1(f)    # 64, 128, 64
        f = self.feats.layer2(f)    # 128, 64, 32
        f = self.feats.layer3(f)    # 256, 64, 32
        f = self.feats.layer4(f)    # 512, 64, 32
        p = self.psp(f)             # 512, 64, 32
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        p = self.final(p)
        p = torch.sigmoid(p)
        # auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
		#self.classifier(auxiliary)
        return p


if __name__ == "__main__":
    input_img = torch.rand((4,1,512,256)).cuda()
    net = PSPNet(num_classes=1,in_channels=1).cuda()
    out = net(input_img)