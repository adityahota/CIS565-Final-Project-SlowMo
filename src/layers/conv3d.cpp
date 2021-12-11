#include "conv3d.h"

Conv3d::Conv3d(std::string filterFile, std::string biasFile, cudnnActivationMode_t actMode,
               Dims5 dims_in, Dims5 dims_filter, Dims3 padding, Dims3 stride, Dims3 dilation)
{
    // Assign all dimensional information
    this->dims_in = dims_in;
    this->dims_filter = dims_filter;
    this->pad = padding;
    this->str = stride;
    this->dil = dilation;

    // Create tensor, filter, and convolution descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnCreateFilterDescriptor(&desc_filter));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&desc_conv));

    // Set input tensor descriptor
    //      First, get memory access strides
    int data_in_stride[5] = {GET_DIM5_C(dims_in) * GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in), GET_DIM5_W(dims_in), 1};
    //      Then, set the tensor descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_in,
        CUDNN_DATA_FLOAT,
        CONV3D_TENSOR_KERN_DIM,
        this->dims_in.dims,
        data_in_stride));

    // Set convolution descriptor
    checkCUDNN(cudnnSetConvolutionNdDescriptor(
        desc_conv,
        CONV3D_TENSOR_KERN_DIM - 2,
        this->pad.dims,
        this->str.dims,
        this->dil.dims,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // Set filter descriptor
    checkCUDNN(cudnnSetFilterNdDescriptor(
        desc_filter,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        CONV3D_TENSOR_KERN_DIM,
        this->dims_filter.dims));

    // Get output tensor dimensions
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
        desc_conv,
        desc_in,
        desc_filter,
        CONV3D_TENSOR_KERN_DIM,
        this->dims_out.dims));

    // Set output descriptor
    //      First, get memory access strides
    int data_out_stride[5] = {GET_DIM5_C(dims_out) * GET_DIM5_D(dims_out) * GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out),
                              GET_DIM5_D(dims_out) * GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out),
                              GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out), GET_DIM5_W(dims_out), 1};
    //      Then, set the tensor descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_out,
        CUDNN_DATA_FLOAT,
        CONV3D_TENSOR_KERN_DIM,
        this->dims_out.dims,
        nullptr));

    // // dev_filter (read file and cudamalloc+cudamemcpy)
    // SizedArrayFloat filtData = readTensor2FloatBuffer(filterFile);
    // cudaMalloc(&dev_filter, filtData.count * sizeof(float));
    // cudaMemcpy(dev_filter, filtData.arr, filtData.count * sizeof(float), cudaMemcpyHostToDevice);
    // delete[] filtData.arr;

    // if constexpr (hasBias)
    // {
    //     // // biasDesc (gotten from filename)
    //     // checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
    //     // // checkCUDNN(cudnnSetTensorNdDescriptor(biasDesc, CUDNN_DATA_FLOAT, ));

    //     // // dev_bias (read file and cudamalloc+cudamemcpy)
    //     // SizedArrayFloat biasData = readTensor2FloatBuffer(biasFile);
    //     // cudaMalloc(&dev_bias, biasData.count * sizeof(float));
    //     // cudaMemcpy(dev_bias, biasData.arr, biasData.count * sizeof(float), cudaMemcpyHostToDevice);
    //     // delete[] biasData.arr;
    // }

    // algo ??
    // workspaceSize (use cudnn to calc)
    // dev_workspace (cudamalloc + cudamemcpy)
}

void Conv3d::run(cudnnHandle_t h, cudnnTensorDescriptor_t const *inputDesc, void *input,
                 cudnnTensorDescriptor_t *outputDesc, void *output, TagUnionExtraRet *extra)
{
    return;
}
