#pragma once
#include "conv3d.h"
#include "avgPool3d.h"

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
     * @param output tensor to store output; passed in; malloc'd by FCLayer, self frees on destruction
     * @param extra ignored
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        // Do not mutate input
        pool->run(h, nullptr, input, nullptr, &postPool, nullptr);
        fcLayer->run(h, nullptr, postPool, nullptr, &postFC, nullptr);
        float flo = 0.f;
        float *floo = &flo;
        s->run(h, &postFCDesc, postFC, nullptr, &floo, nullptr);
        checkCUDNN(cudnnOpTensor(h,
                                 opDesc,
                                 &one, inDescT, input,
                                 &one, postFCDesc, postFC,
                                 &zero, outDescT, output)); // Multiply cudnnOpTensor() output = input * output of sigmoid
    }
    ~Gate()
    {
        cudaFree(postFC);
        cudaFree(postPool);
        // fcLayer->~Conv3dBias();
        // pool->~AvgPool3d();
        delete pool;
        delete fcLayer;
        delete s;
        checkCUDNN(cudnnDestroyTensorDescriptor(postFCDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(inDescT));
        checkCUDNN(cudnnDestroyTensorDescriptor(outDescT));
        checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
    }
    Gate(Dims5 poolDimsIn, Dims3 poolWIn, Dims3 poolPadIn, Dims3 poolStrideIn,
         std::string fcFilter, std::string fcBias,
         Dims5 filtDimsIn, Dims3 filtPad, Dims3 filtStride, Dims3 filtDilation)
    {
        pool = new AvgPool3d(poolDimsIn, poolWIn, poolPadIn, poolStrideIn);
        fcLayer = new Conv3dBias(fcFilter, fcBias, filtDimsIn, filtPad, filtStride, filtDilation);
        s = new Sigmoid();
        checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
        checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));
        cudnnCreateTensorDescriptor(&inDescT);
        cudnnCreateTensorDescriptor(&postFCDesc);
        cudnnCreateTensorDescriptor(&outDescT);
    }

private:
    AvgPool3d *pool;
    Conv3dBias *fcLayer; // conv is fully connected, bias is bias, activation is sigmoid
    Sigmoid *s;
    float *postPool;
    float *postFC;
    cudnnOpTensorDescriptor_t opDesc;
    cudnnTensorDescriptor_t outDescT;
    cudnnTensorDescriptor_t postFCDesc;
    cudnnTensorDescriptor_t inDescT;
};
