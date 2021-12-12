#pragma once
#include "conv3d.h"

class BStem : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        //  l.run() no bias, activation relu
        // l.run(h, nullptr, tensIn, nullptr, tensOut, extraTag);
        l->run(h, &inDescT, input, &outDescT, output, nullptr);
    }

    BStem()
    {
        int tmp1[5] = {1, 3, 4, 256, 448};
        int tmp2[5] = {64, 3, 3, 7, 7};
        int tmp3[3] = {1, 3, 3};
        int tmp4[3] = {1, 2, 2};
        int tmp5[3] = {1, 1, 1};
        Dims5 inputDims = mkDims5(tmp1);
        Dims5 filterDims = mkDims5(tmp2);
        Dims3 paddingDims = mkDims3(tmp3);
        Dims3 strideDims = mkDims3(tmp4);
        Dims3 dilationDims = mkDims3(tmp5);
        l = new Conv3d("module.encoder.stem.0.weight__64x3x3x7x7.bin", "", CUDNN_ACTIVATION_RELU,
                       inputDims, filterDims, paddingDims, strideDims, dilationDims);
    }

    ~BStem()
    {
        l->~Conv3d();
    }

private:
    Conv3d *l;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
};
