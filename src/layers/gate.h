#pragma once
#include "conv3d.h"
#include "avgPool3d.h"
#include "lrelu.h"

/**
 * @brief Class for SEGate: AvgPool FCLayer Sigmoid ProductWithInput
 *
 */
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
     * @param output tensor to store output; passed in; malloc'd by self during construction, caller responsible for freeing
     * @param extra ignored
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;
    ~Gate();
    Gate(Dims5 poolDimsIn, std::string fcFilter, std::string fcBias, cudnnHandle_t h);

    Dims5 getOutputDims();

private:
    AvgPool3d *pool;
    Conv3dBias *fcLayer; // conv is fully connected, bias is bias, activation is sigmoid
    Sigmoid *s;
    float *postPool;
    float *postFC;
    cudnnOpTensorDescriptor_t opDesc;
    cudnnTensorDescriptor_t postFCDesc;
    cudnnTensorDescriptor_t multiplyDesc;
    int sigmoidSize;
    Dims5 gateInputDims;
    Dims5 gateOutputDims;
    float *dev_outBuf;
};
