import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if backbone == 'resnest':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 256
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'resnet':
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class FPN(nn.Module):
    def __init__(self, num_classes, backbone):
        super(FPN, self).__init__()
        if backbone == 'resnest':
            low_feat1 = 256
            low_feat2 = 512
            low_feat3 = 1024
            low_feat4 = 2048
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(nn.Conv2d(low_feat1, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(low_feat2, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(low_feat3, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(low_feat4, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU()
                                   )

        self.Conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU()
                                    )


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, x1, x2, x3, x4):
        p4 = self.conv4(x4)
        l4 = self.Conv(p4)

        p3 = self.conv3(x3)
        p3a = self.upsample(p4) + p3
        l3 = self.Conv(p3a)

        p2 = self.conv2(x2)
        p2a = self.upsample(p3a) + p2
        l2 = self.Conv(p2a)

        p1 = self.conv1(x1)
        p1a = self.upsample(p2a) + p1
        l1 = self.Conv(p1a)

        x = x + l4
        x = self.upsample(x) + l3
        x = self.upsample(x) + l2
        x = self.upsample(x) + l1

        x = self.last_conv(x)

        return x