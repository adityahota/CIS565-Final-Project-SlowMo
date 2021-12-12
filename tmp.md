```
Needed

3dconv
3dTransConv https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/NaiveConvolutionTranspose3d.cu
Relu
Sigmoid
LRelu
2dConv
adaptiveavg3dpool https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/AdaptiveAveragePooling3d.cu

might be relevant https://forums.developer.nvidia.com/t/transposed-convolution/175460/2

FLAVR: 
    input -> unmean -> unet -> postprocess -> remean -> output
    gnarlies: when to change from nchw and back???

unet:
    Stem -> encBlock[4] -> DecBlock[5]

unmean: 
    for (r,g,b) 
        sum for(depth, width, height) for total
    divide by depth*width*height for avg (r,g,b)
    subtract those from tensor
    gnarlies: fast reduce happens in place and destructs the data; might need to duplicate the data

remean:
    add back avg (rgb) for the depth*width*height tensor of each channel

```
