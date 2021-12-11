# Modified from https://github.com/pytorch/vision/tree/master/torchvision/models/video

import torch
import torch.nn as nn

class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""
    def __init__(self, useBias=False):
        super().__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=useBias),
            nn.ReLU(inplace=False)
        )


class SEGating(nn.Module):
    def __init__(self , inplanes , reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, useBias=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=1, bias=useBias),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=useBias),
        )
        self.fg = SEGating(planes) ## Feature Gating
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False) \
                if downsample else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, useBias=False):
        super(Encoder, self).__init__()
        self.inplanes = 64

        self.stem = BasicStem(useBias)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, stride=(1,1,1), downsample=False, useBias=useBias),
                BasicBlock(64, 64, useBias=useBias))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=(1,2,2), downsample=True, useBias=useBias),
                BasicBlock(128, 128, useBias=useBias))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=(1,2,2), downsample=True, useBias=useBias),
                BasicBlock(256, 256, useBias=useBias))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=(1,1,1), downsample=True, useBias=useBias),
                BasicBlock(512, 512, useBias=useBias))


    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0 , x_1 , x_2 , x_3 , x_4

