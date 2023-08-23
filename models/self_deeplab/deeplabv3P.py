from typing import Optional, Union
from torchsummary import summary
from torch import nn
from models.self_deeplab.backbone.ResNest.resnest import *
from models.self_deeplab.ASPP import ASPP, ASPP1
from models.self_deeplab.decoder import *
from models.self_deeplab.backbone.xception import xception
from segmentation_models_pytorch.base.modules import Activation
from models.self_deeplab.backbone.resnet import ResNet101


class Deeplab(nn.Module):
    def __init__(self, num_classes, backbone='resnest', encoder_weights=None, downsample_factor=8, activation: Optional[Union[str, callable]] = None,):
        super(Deeplab, self).__init__()
        if backbone == 'resnest':
            self.backbone = resnest101(pretrained=encoder_weights)
        elif backbone == 'xception':
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=encoder_weights)
        elif backbone == 'resnet':
            self.backbone = ResNet101(downsample_factor=downsample_factor, pretrained=encoder_weights)

        else:
            raise ValueError('Unsupported backbone - `{}`, Use xception, resnest, resnet.'.format(backbone))

        self.aspp = ASPP(backbone, downsample_factor=16)
        # self.aspp = ASPP1(dim_in=2048, dim_out=256)
        self.decoder = Decoder(num_classes, backbone)
        self.activation = Activation(activation)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, x1 = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, x1)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.activation(x)

        return x

if __name__ == "__main__":
    model = Deeplab(backbone='resnest', num_classes=21, encoder_weights=True)
    print(model)
    model.cuda()
    input = torch.rand(4, 3, 224, 224)
    output = model(input)
    print(output.size())

