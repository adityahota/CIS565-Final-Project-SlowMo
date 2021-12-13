import os
import torch
import torch.nn as nn
import time
import numpy as np


class SEGating(nn.Module):
    def __init__(self , inplanes , reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):                       #### (1,64,4,128,224) -> (1,64,4,128,224)
        out = self.pool(x)                      #### (1,64,4,128,224) -> (1,64,1,1,1)
        y = self.attn_layer(out)                #### (1,64,1,1,1) -> (1,64,1,1,1)
        return x * y                            #### (1,64,4,128,224)*(1,64,1,1,1) -> (1,64,4,128,224)

#####
##### Encoder Classes
#####

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
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, useBias=False):
        super(Encoder, self).__init__()

        self.stem = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=useBias),
                nn.ReLU(inplace=False)
        )

        self.layer1_1 = BasicBlock(64, 64, stride=(1,1,1), downsample=False, useBias=useBias)
        self.layer1_2 = BasicBlock(64, 64, useBias=useBias)

        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=(1,2,2), downsample=True, useBias=useBias),
                BasicBlock(128, 128, useBias=useBias))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=(1,2,2), downsample=True, useBias=useBias),
                BasicBlock(256, 256, useBias=useBias))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=(1,1,1), downsample=True, useBias=useBias),
                BasicBlock(512, 512, useBias=useBias))

    def forward(self, x):
        x_0 = self.stem(x)
        x_1 = self.layer1_1(x_0)
        x_1 = self.layer1_2(x_1)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_0, x_1, x_2, x_3, x_4



########
######## Decoder Classes
########

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

        out_channels = 3 * n_outputs

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.encoder = Encoder(useBias=(n_outputs > 1))

        self.decoder = nn.Sequential(
            Conv_3d(512, 256, kernel_size=3, padding=1, bias=True),
            upConv3D(512, 128, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            upConv3D(256, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            Conv_3d(128, 64 , kernel_size=3, padding=1, bias=True),
            upConv3D(128, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))
        )

        self.feature_fuse = nn.Conv2d(64 * n_inputs, 64, kernel_size=1, stride=1, padding=0, bias=False)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels , kernel_size=7 , stride=1, padding=0) 
        )         

    def forward(self, images):
        _, _, H, W = images[0].shape
        print(H, W)
        images = torch.stack(images, dim=2)

        ## Batch mean normalization works slightly better than global mean normalization
        ## https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mean_ 
        a, b, c, d, e = images.shape
        images.cpu().numpy().tofile(f"encoder-input__{a}x{b}x{c}x{d}x{e}.bin")

        # x_0 , x_1, x_2 , x_3 , x_4 = self.encoder(images)
        x_0 , _x_1, _x_2 , _x_3 , _x_4 = self.encoder(images)
        os.system(f"./CUDA-Convolution2D ./encoder-input__{a}x{b}x{c}x{d}x{e}.bin")

        # x_0 = torch.Tensor(np.fromfile("x0_cudnn.bin", dtype=np.float32).reshape(1,64,4,H//2,W//2)).cuda()
    #    print("GOT IT: ", x_0.shape)
        x_1 = torch.Tensor(np.fromfile("x1_cudnn.bin", dtype=np.float32).reshape(1,64,4,H//2,W//2)).cuda()
    #    print("GOT IT: ", x_1.shape)
        x_2 = torch.Tensor(np.fromfile("x2_cudnn.bin", dtype=np.float32).reshape(1,128,4,H//4,W//4)).cuda()
    #    print("GOT IT: ", x_2.shape)
        x_3 = torch.Tensor(np.fromfile("x3_cudnn.bin", dtype=np.float32).reshape(1,256,4,H//8,W//8)).cuda()
    #    print("GOT IT: ", x_3.shape)
        x_4 = torch.Tensor(np.fromfile("x4_cudnn.bin", dtype=np.float32).reshape(1,512,4,H//8,W//8)).cuda()
    #    print("GOT IT: ", x_4.shape)


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

