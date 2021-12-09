#pragma once
#include "gate.h"
#include "lrelu.h"

class DecBlock : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        // deConv.run()
        // g.run()
        // lr.run()
    }

private:
    Conv3d deConv;
    Gate g;
    LReLU lr;
};
