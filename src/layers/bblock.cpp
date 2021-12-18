#include "bblock.h"

BBlock::BBlock(Dims5 blockDimsIn,
               std::string conv1_weights, Dims3 conv1_str,
               std::string conv2_weights, Dims3 conv2_str,
               std::string fg_weights, std::string fg_bias,
               bool downsample, std::string downsample_weights, Dims3 downsample_stride, cudnnHandle_t h)
{
    // Store whether or not downsampling is required
    downsampleFlag = downsample;

    // Conv1 is a Conv3d followed by a ReLU
    //      Conv3d
    Dims3 conv1_convPadding = unitDims3;
    Dims3 conv1_convStride = conv1_str;
    Dims3 conv1_convDilate = unitDims3;
    conv1_conv = new Conv3d(conv1_weights, blockDimsIn,
                            conv1_convPadding, conv1_convStride, conv1_convDilate, h);

    //      ReLU
    relu = new LReLU();

    // Conv2 is just a Conv3d
    Dims5 conv1_dimsOut = conv1_conv->getOutputDim();
    Dims3 conv2_convPadding = unitDims3;
    Dims3 conv2_convStride = conv2_str;
    Dims3 conv2_convDilate = unitDims3;
    conv2_conv = new Conv3d(conv2_weights, conv1_dimsOut,
                            conv2_convPadding, conv2_convStride, conv2_convDilate, h);

    // Gating layer (fg) is a gate
    Dims5 conv2_dimsOut = conv2_conv->getOutputDim();
    g = new Gate(conv2_dimsOut, fg_weights, fg_bias, h);

    // Remember to call ReLU

    // Downsample layer (if required)
    if (downsampleFlag)
    {
        Dims3 downsample_padding = zeroDims3;
        Dims3 downsample_dilate = unitDims3;
        this->downsample = new Conv3d(downsample_weights, blockDimsIn, downsample_padding, downsample_stride, downsample_dilate, h);
    }

    // Calculate output dimensions here
    // TODO: check
    blockDimsOut = g->getOutputDims();

    cudnnCreateTensorDescriptor(&postGateDesc);
    cudnnSetTensorNdDescriptor(postGateDesc, CUDNN_DATA_FLOAT, 5,
                               conv2_dimsOut.dims, dim5Stride(conv2_dimsOut).dims);
    cudnnCreateTensorDescriptor(&inDescT);
    cudnnSetTensorNdDescriptor(inDescT, CUDNN_DATA_FLOAT, 5,
                               blockDimsIn.dims, dim5Stride(blockDimsIn).dims);
    relu1Size = dims5ToSize(conv1_dimsOut);
    relu2Size = downsampleFlag ? dims5ToSize(this->downsample->getOutputDim())
                               : dims5ToSize(blockDimsIn); // may depend on downsampel
    if (downsampleFlag)
    {
        cudnnCreateTensorDescriptor(&dev_outBufDesc);
        cudnnSetTensorNdDescriptor(dev_outBufDesc, CUDNN_DATA_FLOAT, 5,
                                   this->downsample->getOutputDim().dims,
                                   dim5Stride(this->downsample->getOutputDim()).dims);
    }
}

void BBlock::run(cudnnHandle_t h,
                 cudnnTensorDescriptor_t const *inputDesc, float *input,
                 cudnnTensorDescriptor_t *outputDesc, float **output,
                 TagUnionExtraRet *extra)
{
    std::cout << "Running Basic Block..." << std::endl;

    if (downsampleFlag)
    {
        conv1_conv->run(h, nullptr, input, nullptr, &postConv1, nullptr);
        relu->run(relu1Size, postConv1);
        conv2_conv->run(h, nullptr, postConv1, nullptr, &postConv2, nullptr);
        g->run(h, nullptr, postConv2, nullptr, &postGate, nullptr);
        downsample->run(h, nullptr, input, nullptr, &dev_outBuf, nullptr);
        cudnnAddTensor(h,
                       &one, postGateDesc, postGate,
                       &one, dev_outBufDesc, dev_outBuf); // A is the output so far, c is initial input
        relu->run(relu2Size, dev_outBuf);
        *output = dev_outBuf;
    }
    else
    {
        conv1_conv->run(h, nullptr, input, nullptr, &postConv1, nullptr);

        relu->run(relu1Size, postConv1);

        conv2_conv->run(h, nullptr, postConv1, nullptr, &postConv2, nullptr);

        g->run(h, nullptr, postConv2, nullptr, &postGate, nullptr);

        cudnnAddTensor(h,
                       &one, postGateDesc, postGate,
                       &one, inDescT, input); // A is the output so far, c is initial input

        relu->run(relu2Size, input);
        *output = input;
    }
}

BBlock::~BBlock()
{
    cudnnDestroyTensorDescriptor(postGateDesc);
    cudnnDestroyTensorDescriptor(inDescT);
    cudaFree(postConv1);
    cudaFree(postConv2);
    cudaFree(postGate);
    if (downsampleFlag)
    {
        cudaFree(dev_outBuf);
        cudnnDestroyTensorDescriptor(dev_outBufDesc);
    }
}

Dims5 BBlock::getOutputDims()
{
    return blockDimsOut;
}
