#pragma once
#include "conv3d.h"
#include "avgPool3d.h"

class Gate : Runnable
{
public:
    /**
     * @brief Concrete run implementation
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
             cudnnTensorDescriptor_t *outputDesc, void **output,
             TagUnionExtraRet *extra) override
    {
        // Do not mutate input
        pool.run(h, nullptr, input, nullptr, postPool, nullptr);
        fcLayer.run(h, nullptr, postPool, nullptr, postFC, nullptr);
        // Multiply cudnnOpTensor() output = input * output of sigmoid
        checkCUDNN(cudnnOpTensor(h,
                                 opDesc,
                                 &one, inDescT, input,
                                 &one, postFCDesc, postFC,
                                 &zero, outDescT, output));
    }

private:
    AvgPool3d pool;
    Conv3dBias fcLayer; // conv is fully connected, bias is bias, activation is sigmoid
    float *postPool;
    float *postFC;
    cudnnOpTensorDescriptor_t opDesc;
    cudnnTensorDescriptor_t outDescT;
    cudnnTensorDescriptor_t postFCDesc;
    cudnnTensorDescriptor_t inDescT;
};
