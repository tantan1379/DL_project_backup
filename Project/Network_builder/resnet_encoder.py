import torch
import torch.nn as nn
from torchvision import models

class resnet_encoder(nn.Module):
    def __init__(self):
        super(resnet_encoder, self).__init__()
        self.backbone = models.resnet34(pretrained=False)
    
    def forward(self, x):
        batch_size = x.size(0)
        print(x.shape)
        x = self.backbone.conv1(x)
        print(x.shape)
        x = self.backbone.bn1(x)
        print(x.shape)
        c1 = self.backbone.relu(x)
   
        x = self.backbone.maxpool(c1)
        print(x.shape)     
        c2 = self.backbone.layer1(x)
        print(c2.shape)
        c3 = self.backbone.layer2(c2)
        print(c3.shape)
        c4 = self.backbone.layer3(c3)
        print(c4.shape)
        c5 = self.backbone.layer4(c4)
        print(c5.shape)
        
        return c5

if __name__ == '__main__':
    model = resnet_encoder().cuda()
    input_img = torch.Tensor(1,3,256,128).cuda()
    output = model(input_img)
    # print(output.shape)