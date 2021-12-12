#pragma once

#include "runnable.h"

class AvgPool3d : Runnable
{
public:
    /**
     * @brief Concrete run implementation; input and output must not overlap
     *
     * @param h handle
     * @param inputDesc ignored; descriptors set at construction time
     * @param input input float tensor; passed in; self never owns
     * @param outputDesc ignored; descriptors set at construction time
     * @param output tensor to store output; passed in; self mallocs, caller responsible for freeing
     * @param extra ignored
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        cudaMalloc(&dev_out, size * sizeof(float));
        *output = dev_out;
        checkCUDNN(cudnnPoolingForward(h, poolDesc,
                                       &one, inDescT, input,
                                       &zero, outDescT, output));
    }

private:
    int size;
    float *dev_out;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
};
