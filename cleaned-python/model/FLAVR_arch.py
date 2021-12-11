import importlib

import torch
import torch.nn as nn
from .resnet_3D import SEGating


class Conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)

class upConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            SEGating(out_ch)
        )
   
    def forward(self, x):
        return self.upconv(x)

class Conv_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            SEGating(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        nf = [512 , 256 , 128 , 64]
        out_channels = 3*n_outputs

        growth = 2
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module(".resnet_3D" , "model")
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , "unet_18")()#FIXME

        self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[1] , kernel_size=3, padding=1, bias=True),
            upConv3D(nf[1] * 2, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            upConv3D(nf[2] * 2, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            Conv_3d(nf[3] * 2, nf[3] , kernel_size=3, padding=1, bias=True),
            upConv3D(nf[3] * 2, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))
        )

        self.feature_fuse = Conv_2d(nf[3]*n_inputs, nf[3], kernel_size=1, stride=1)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels , kernel_size=7 , stride=1, padding=0) 
        )         

    def forward(self, images):

        images = torch.stack(images , dim=2)

        ## Batch mean normalization works slightly better than global mean normalization
        ## https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mean_ 

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3 , x_3] , dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2 , x_2] , dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1 , x_1] , dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0 , x_0] , dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        out = torch.split(out, dim=1, split_size_or_sections=3)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
 
        return out

