#pragma once

#include "runnable.h"

#define Conv3d_TENSOR_KERN_DIM 5
#define CONV2D_TENSOR_KERN_DIM 4

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
     * @param output tensor to store output; passed in; pool mallocs, caller responsible for freeing
     * @param extra ignored
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;

    AvgPool3d(Dims5 dims_in, Dims3 win, Dims3 pad, Dims3 str);

    ~AvgPool3d()
    {
        cudnnDestroyTensorDescriptor(desc_in);
        cudnnDestroyTensorDescriptor(desc_out);
        cudnnDestroyPoolingDescriptor(desc_pool);
        cudaFree(dev_output);
    }

    Dims5 getOutputDim();

private:
    // Descriptors
    cudnnTensorDescriptor_t desc_in;
    cudnnTensorDescriptor_t desc_out;
    cudnnPoolingDescriptor_t desc_pool;

    // Tensor dimensions
    Dims5 dims_in;
    Dims5 dims_out;

    // Pooling parameters (window, padding, stride)
    Dims3 win;
    Dims3 pad;
    Dims3 str;

    float *dev_output;
};
