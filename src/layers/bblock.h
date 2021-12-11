#pragma once
#include "gate.h"

class BBlock : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        // l1.run() conv relu
        // l2.run() conv ident
        // g.run()
        // maybe downsample the initial input
        // cudnnAddTensor() input and current output to output
        // cudnnActivationForward relu inplace on output (no convolution)
    }

private:
    Conv3d l1, l2;
    Gate g;
    bool downsample; //?????
};
