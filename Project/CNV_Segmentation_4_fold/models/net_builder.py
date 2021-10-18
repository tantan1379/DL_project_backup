import sys
sys.path.append("../")

from models.nets.Backbone import Backbone
from models.nets.ResUNet import ResUNet_add,ResUNet_concat
from models.nets.CPFNet import CPFNet
from models.nets.PSPNet import PSPNet
from models.nets.CE_Net import CE_Net,CE_Net_backbone_DAC_without_atrous
from models.nets.DeepLabv3_plus import DeepLabv3_plus


def net_builder(name,in_channels=3,n_class=1):
    print("loading",name)
    if name == 'backbone':
        net = Backbone(in_channels=in_channels,num_classes=n_class)
    elif name == 'unet':
        from models.nets.UNet import UNet
        net = UNet(in_channels=in_channels, n_classes=n_class, feature_scale=2)
    elif name == 'unet_se':
        from models.nets.UNet import UNet_SE
        net = UNet_SE(in_channels=in_channels, n_classes=n_class, feature_scale=2)
    elif name == 'unet_add':
        from models.nets.UNet import UNet_add
        net = UNet_add(in_channels=in_channels, n_classes=n_class)
    elif name == 'unet_spp_se':
        from models.nets.Unet_spp_se import UNet_SPP_SE
        net = UNet_SPP_SE(in_channels=in_channels,n_classes=n_class,feature_scale=2)
    elif name == 'cpfnet':
        net = CPFNet(in_channels=in_channels, out_planes=n_class)
    elif name == 'cenet':
        net = CE_Net(in_channels=in_channels, num_classes=n_class)
    elif name == 'cenet_without_atrous':
        net = CE_Net_backbone_DAC_without_atrous(num_channels=in_channels, num_classes=n_class)
    elif name == "pspnet":
        net = PSPNet(in_channels=in_channels,num_classes=n_class)
    elif name == 'resunet_concat':
        net = ResUNet_concat(in_channels=in_channels,num_classes=n_class)
    elif name == 'resunet_add':
        net = ResUNet_add(in_channels=in_channels,num_classes=n_class)
    elif name == 'deeplabv3plus':
        net = DeepLabv3_plus(nInputChannels=in_channels, n_classes=n_class, pretrained=False)



    else:
        raise NameError("Unknow Model Name!")
    return net


if __name__ == "__main__":

    model_name = 'pspnet'
    model = net_builder(model_name,1,1)
    print('parameters_number = {}'.format(sum(p.numel() for p in list(model.parameters()) if p.requires_grad)))