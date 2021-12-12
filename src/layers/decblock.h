#pragma once
#include "gate.h"
#include "lrelu.h"

class DecBlock : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        // deConv.run()
        // g.run()
        // lr.run()
    }

private:
    Conv3dBias deConv;
    Gate g;
    LReLU lr;
};
