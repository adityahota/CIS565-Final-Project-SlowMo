# Modified from https://github.com/pytorch/vision/tree/master/torchvision/models/video

import torch
import torch.nn as nn

__all__ = ['unet_18']

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1,
                 useBias=False):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=useBias)


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
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            Conv3DSimple(inplanes, planes, midplanes, stride),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Conv3DSimple(planes, planes, midplanes),
        )
        self.fg = SEGating(planes) ## Feature Gating
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VideoResNet(nn.Module):
    def __init__(self, useBias=False):
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = BasicStem(useBias)
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=1)


    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0 , x_1 , x_2 , x_3 , x_4

    def _make_layer(self, planes, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            ds_stride = (1, stride, stride)
            downsample = nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=ds_stride, bias=False)
            stride = ds_stride

        b1 = BasicBlock(self.inplanes, planes, stride, downsample)
        self.inplanes = planes
        b2 = BasicBlock(self.inplanes, planes)

        return nn.Sequential(b1, b2)

def unet_18(useBias=False):
    """
    Construct 18 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Returns:
        nn.Module: R3D-18 encoder
    """

    return VideoResNet(useBias)
