#pragma once

#include "runnable.h"

const int blockSize = 256;

class LReLU : Runnable
{
public:
    // just wraps leaky relu cuda kernel in place, tag union store lrelu info
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override;

private:
    float coeff;
};
