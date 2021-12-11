#pragma once
#include "conv3d.h"

class Gate : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        // Do not mutate input
        // Pool cudnnPoolingForward()
        // fcLayer.run(); conv is fully connected, bias is bias, activation is ident
        // separate sigmoid activation.run()
        // Multiply cudnnOpTensor() output = input * output of sigmoid
    }

private:
    Conv3d<true> fcLayer;
};
