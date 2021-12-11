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
    }

private:
    Conv3d<false> l;
};
