#pragma once
#include "gate.h"
#include "runnable.h"

/**
 * @brief Class that is the Basic Block: conv relu conv gate downsample? addResidual relu
 *
 */
class BBlock : Runnable
{
public:
    /**
     * @brief
     *
     * @param h
     * @param inputDesc
     * @param input
     * @param outputDesc
     * @param output if no downsample, in place on input. if downsample, self mallocs and caller must free
     * @param extra
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;

    // BBlock(Dims5 blockDimsIn,
    //        std::string conv1_weights, Dims3 conv1_str,
    //        std::string conv2_weights, Dims3 conv2_str,
    //        std::string fg_weights, std::string fg_bias, Dims3 fg_str, Dims3 fg_pad,
    //        bool downsample, std::string downsample_weights);

    BBlock(Dims5 blockDimsIn,
           std::string conv1_weights, Dims3 conv1_str,
           std::string conv2_weights, Dims3 conv2_str,
           std::string fg_weights, std::string fg_bias, Dims3 fg_str, Dims3 fg_pad,
           bool downsample, std::string downsample_weights, Dims3 downsample_stride);
    ~BBlock();

    Dims5 getOutputDims();

private:
    Conv3d *conv1_conv, *conv2_conv, *downsample;
    Gate *g;
    LReLU *relu;
    bool downsampleFlag;
    int relu1Size;
    int relu2Size;

    float *dev_outBuf; // only might be used by downsampel

    float *postConv1;                              // set by called fucntions; ignore in ctor
    float *postConv2;                              // set by called fucntions; ignore in ctor
    float *postGate;                               // set by called fucntions; ignore in ctor
    cudnnTensorDescriptor_t postGateDesc, inDescT; // init in ctor
    cudnnTensorDescriptor_t dev_outBufDesc;
    Dims5 blockDimsOut;
};
