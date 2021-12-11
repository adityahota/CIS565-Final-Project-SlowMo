# Modified from https://github.com/pytorch/vision/tree/master/torchvision/models/video

import torch
import torch.nn as nn

__all__ = ['unet_18']

useBias = False

class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=useBias)


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""
    def __init__(self):
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
        
    def forward(self , x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
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
    def __init__(self, block, conv_makers, layers,
                 stem, zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1 )
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2 , temporal_stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2 , temporal_stride=1)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=1, temporal_stride=1)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0 , x_1 , x_2 , x_3 , x_4

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            ds_stride = (temporal_stride if temporal_stride else stride, stride, stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes,
                          kernel_size=1, stride=ds_stride, bias=False),
            )
            stride = ds_stride

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample ))

        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder ))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def unet_18():
    """
    Construct 18 layer Unet3D model as in
    https://arxiv.org/abs/1711.11248

    Returns:
        nn.Module: R3D-18 encoder
    """

    return VideoResNet(block=BasicBlock,
            conv_makers=[Conv3DSimple] * 4,
            layers=[2, 2, 2, 2],
            stem=BasicStem)
