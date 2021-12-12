#include "avgPool3d.h"

AvgPool3d::AvgPool3d(Dims5 dims_in, Dims3 win, Dims3 pad, Dims3 str)
{
    this->dims_in = dims_in;
    this->win = win;
    this->pad = pad;
    this->str = str;

    // Create tensor and pooling descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnCreatePoolingDescriptor(&desc_pool));

    // Set input tensor descriptor
    int data_in_stride[5] = {GET_DIM5_C(dims_in) * GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in), GET_DIM5_W(dims_in), 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_in,
        CUDNN_DATA_FLOAT,
        CONV3D_TENSOR_KERN_DIM,
        this->dims_in.dims,
        data_in_stride));

    // Set pooling descriptor
    checkCUDNN(cudnnSetPoolingNdDescriptor(
        desc_pool,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        CONV3D_TENSOR_KERN_DIM - 2,
        win.dims,
        pad.dims,
        str.dims));

    // Get output tensor dimensions
    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(
        desc_pool,
        desc_in,
        CONV3D_TENSOR_KERN_DIM,
        dims_out.dims));

    // Set output tensor descriptor
    int data_out_stride[5] = {GET_DIM5_C(dims_out) * GET_DIM5_D(dims_out) * GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out),
                              GET_DIM5_D(dims_out) * GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out),
                              GET_DIM5_H(dims_out) * GET_DIM5_W(dims_out), GET_DIM5_W(dims_out), 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_out,
        CUDNN_DATA_FLOAT,
        CONV3D_TENSOR_KERN_DIM,
        dims_out.dims,
        data_out_stride));
}

void AvgPool3d::run(cudnnHandle_t h,
                    cudnnTensorDescriptor_t const *inputDesc, float *input,
                    cudnnTensorDescriptor_t *outputDesc, float **output,
                    TagUnionExtraRet *extra)
{
    // Allocate space on GPU for output tensor
    int num_elements_out = dims_out.dims[0] * dims_out.dims[1] * dims_out.dims[2] * dims_out.dims[3] * dims_out.dims[4];
    cudaMalloc(output, num_elements_out * sizeof(float));
    cudaMemset(*output, 0, num_elements_out * sizeof(float));

    // Run the pooling operation
    checkCUDNN(cudnnPoolingForward(
        h,
        desc_pool,
        &one,
        desc_in,
        input,
        &zero,
        desc_out,
        *output));
}

Dims5 AvgPool3d::getOutputDim()
{
    return dims_out;
}