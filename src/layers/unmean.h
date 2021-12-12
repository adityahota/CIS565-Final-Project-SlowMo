#pragma once
#include "runnable.h"

class UnMean : Runnable
{
public:
    /**
     * @brief Finds the RGB Means of the input frames and normalizes them
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output
     * @param output start of the ouput tensor data
     * @param extra passes out rgb averages for the ReMean step
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void **output,
             TagUnionExtraRet *extra) override;

    UnMean(); // TODO;
    ~UnMean();

private:
    cudnnReduceTensorDescriptor_t tensReduceDescT; // CUDNN_REDUCE_TENSOR_AVG
    void *wkspc;
    size_t wkspcSize;
    cudnnTensorDescriptor_t inDescT; // dim 1x3x4xHxW
    void *rgbAvgs;
    cudnnTensorDescriptor_t rgbAvgsDescT;  // dim 1x3x1x1x1
    cudnnOpTensorDescriptor_t tensOpDescT; // CUDNN_OP_TENSOR_ADD
    cudnnTensorDescriptor_t outDescT;      // dim 1x3x4xHxW
    TagUnionExtraRet extras;
};
