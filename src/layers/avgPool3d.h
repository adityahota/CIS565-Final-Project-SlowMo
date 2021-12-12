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
     * @param output tensor to store output; passed in; self never owns
     * @param extra ignored
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        checkCUDNN(cudnnPoolingForward(h, poolDesc,
                                       &one, inDescT, input,
                                       &zero, outDescT, output));
    }

private:
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
};
