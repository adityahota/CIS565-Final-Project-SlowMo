import torch
import torch.nn as nn


class SEGating(nn.Module):
    def __init__(self , inplanes , reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True)
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        print("\nSEGating in:\t", x.shape)
        y = self.pool(x)
        print("SEGating pool:\t", y.shape)
        y = self.conv(y)
        print("SEGating conv:\t", y.shape)
        y = self.sigm(y)
        print("SEGating sigm:\t", y.shape)
        print("SEGating x*y:\t", (x*y).shape)
        return x * y

#####
##### Encoder Classes
#####

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, useBias=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=1, bias=useBias)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=useBias)
        
        self.fg = SEGating(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False) \
                if downsample else None

    def forward(self, x):
        print("\nBlock in:\t", x.shape)
        out = self.conv1(x)
        print("Block conv1:\t", out.shape)
        out = self.relu(out)
        print("Block relu:\t", out.shape)
        out = self.conv2(out)
        print("Block conv2:\t", out.shape)
        out = self.fg(out)
        print("Block fg:\t", out.shape)
        if self.downsample is not None:
            x = self.downsample(x)
            print("Block down:\t", x.shape)
        out += x
        print("Block out+x:\t", out.shape)
        print("Block relu:\t", self.relu(out).shape)
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, useBias=False):
        super(Encoder, self).__init__()

        self.stem_conv = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=useBias)
        self.stem_relu = nn.ReLU(inplace=False)

        self.layer1_1 = BasicBlock(64, 64, stride=(1,1,1), downsample=False, useBias=useBias)
        self.layer1_2 = BasicBlock(64, 64, useBias=useBias)

        self.layer2_1 = BasicBlock(64, 128, stride=(1,2,2), downsample=True, useBias=useBias)
        self.layer2_2 = BasicBlock(128, 128, useBias=useBias)

        self.layer3_1 = BasicBlock(128, 256, stride=(1,2,2), downsample=True, useBias=useBias)
        self.layer3_2 = BasicBlock(256, 256, useBias=useBias)

        self.layer4_1 = BasicBlock(256, 512, stride=(1,1,1), downsample=True, useBias=useBias)
        self.layer4_2 = BasicBlock(512, 512, useBias=useBias)

    def forward(self, x):
        print("\n\nEncoder in:\t", x.shape)
        x_0 = self.stem_conv(x)
        print("Encoder stem_conv:\t", x_0.shape)
        x_0 = self.stem_relu(x_0)
        print("Encoder x_0:\t", x_0.shape)

        x_1 = self.layer1_2(x_0)
        print("Encoder layer1_1:\t", x_1.shape)
        x_1 = self.layer1_2(x_1)
        print("Encoder x_1:\t", x_1.shape)
        
        x_2 = self.layer2_1(x_1)
        print("Encoder layer2_1:\t", x_2.shape)
        x_2 = self.layer2_2(x_2)
        print("Encoder x_2:\t", x_2.shape)

        x_3 = self.layer3_1(x_2)
        print("Encoder layer3_1:\t", x_3.shape)
        x_3 = self.layer3_2(x_3)
        print("Encoder x_2:\t", x_3.shape)

        x_4 = self.layer4_1(x_3)
        print("Encoder layer4_1:\t", x_4.shape)
        x_4 = self.layer4_2(x_4)
        print("Encoder x_4:\t", x_4.shape)
        return x_0, x_1, x_2, x_3, x_4



########
######## Decoder Classes
########

class upConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = SEGating(out_ch)
   
    def forward(self, x):
        print("upConv3D in: ", x.shape)
        x = self.conv(x)
        print("upConv3D conv: ", x.shape)
        x = self.gate(x)
        print("upConv3D gate: ", x.shape)
        return x

class Conv_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        self.gate = SEGating(out_ch)

    def forward(self, x):
        print("Conv_3d in: ", x.shape)
        x = self.conv(x)
        print("Conv_3d conv: ", x.shape)
        x = self.gate(x)
        print("Conv_3d gate: ", x.shape)
        return x





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
        print("\nUNet in: ", images[0].shape)

        images = torch.stack(images, dim=2)
        print("UNet stacked images:", images.shape)

        ## Batch mean normalization works slightly better than global mean normalization
        ## https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mean_ 

        print("UNet normalized images", images.shape)

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        print("UNet encoder output x_0:", x_0.shape)
        print("UNet encoder output x_1:", x_1.shape)
        print("UNet encoder output x_2:", x_2.shape)
        print("UNet encoder output x_3:", x_3.shape)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        print("UNet lrelu dx_3:", dx_3.shape)
        dx_3 = torch.cat([dx_3 , x_3] , dim=1)
        print("UNet cat dx_3:", dx_3.shape)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        print("UNet lrelu dx_2:", dx_2.shape)
        dx_2 = torch.cat([dx_2 , x_2] , dim=1)
        print("UNet cat dx_2:", dx_2.shape)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        print("UNet lrelu dx_1:", dx_1.shape)
        dx_1 = torch.cat([dx_1 , x_1] , dim=1)
        print("UNet cat dx_1:", dx_1.shape)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        print("UNet lrelu dx_0:", dx_0.shape)
        dx_0 = torch.cat([dx_0 , x_0] , dim=1)
        print("UNet cat dx_0:", dx_0.shape)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        print("UNet lrelu dx_out:", dx_out.shape)
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)
        print("UNet cat dx_out:", dx_out.shape)

        out = self.lrelu(self.feature_fuse(dx_out))
        print("UNet lrelu out:", out.shape)
        out = self.outconv(out)
        print("UNet outconv out:", out.shape)

        out = torch.split(out, dim=1, split_size_or_sections=3)
        print("UNet spluit out:", out.shape)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]

        print("UNet squeezed out[0]:", out[0].shape)

        return out

