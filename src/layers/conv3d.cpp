#include "conv3d.h"

Conv3d::Conv3d(std::string filterFile, Dims5 dims_in,
               Dims3 padding, Dims3 stride, Dims3 dilation)
{
    // Assign all dimensional information
    this->dims_in = dims_in;
    this->pad = padding;
    this->str = stride;
    this->dil = dilation;

    // Obtain filter dimensions from file
    // TsrDims tsr_dims_filter = filename2dims(filterFile);
    this->dims_filter = filename2dims5(filterFile);

    // Load filter from file
    SizedArrayFloat filter_weights = readTensor2FloatBuffer(filterFile);

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
        Conv3d_TENSOR_KERN_DIM,
        this->dims_in.dims,
        data_in_stride));

    // Set convolution descriptor
    checkCUDNN(cudnnSetConvolutionNdDescriptor(
        desc_conv,
        Conv3d_TENSOR_KERN_DIM - 2,
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
        Conv3d_TENSOR_KERN_DIM,
        this->dims_filter.dims));

    // Get output tensor dimensions
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
        desc_conv,
        desc_in,
        desc_filter,
        Conv3d_TENSOR_KERN_DIM,
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
        Conv3d_TENSOR_KERN_DIM,
        this->dims_out.dims,
        data_out_stride));

    // Allocate space on GPU for filter and delete from host
    cudaMalloc(&dev_filter, filter_weights.count * sizeof(float));
    cudaMemcpy(dev_filter, filter_weights.arr, filter_weights.count * sizeof(float), cudaMemcpyHostToDevice);
    delete[] filter_weights.arr;
}

void Conv3d::run(cudnnHandle_t h, cudnnTensorDescriptor_t const *inputDesc, float *input,
                 cudnnTensorDescriptor_t *outputDesc, float **output, TagUnionExtraRet *extra)
{
    // checkCUDAError("cuda pree");
    // Allocate space on GPU for output tensor
    int num_elements_out = dims_out.dims[0] * dims_out.dims[1] * dims_out.dims[2] * dims_out.dims[3] * dims_out.dims[4];
    float *dev_tmp;
    static bool reached = false;
    //    if (reached)
    //    {
    //        cudaFree(input);
    //    }
    //    if (!reached)
    //    {
    //    	reached = true;
    //    }
    cudaMalloc(&dev_tmp, num_elements_out * sizeof(float));
    // checkCUDAError("cuda mallo amgory");
    cudaMemset(dev_tmp, 0, num_elements_out * sizeof(float));
    // checkCUDAError("nvida ree");

    // Initialize the algorithm
    cudnnConvolutionFwdAlgoPerf_t algorithm_perf;
    int returned_algorithms = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        h,
        desc_in,
        desc_filter,
        desc_conv,
        desc_out,
        1,
        &returned_algorithms,
        &algorithm_perf));

    // Allocate workspace
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        h,
        desc_in,
        desc_filter,
        desc_conv,
        desc_out,
        algorithm_perf.algo,
        &dev_workspace_bytes));
    cudaMalloc(&dev_workspace, dev_workspace_bytes);

    // Run the convolution
    checkCUDNN(cudnnConvolutionForward(
        h,
        &one, desc_in, input,
        desc_filter, dev_filter,
        desc_conv, algorithm_perf.algo,
        dev_workspace, dev_workspace_bytes,
        &zero, desc_out, dev_tmp));
    *output = dev_tmp;

    return;
}

Dims5 Conv3d::getOutputDim()
{
    return dims_out;
}

Conv3dBias::Conv3dBias(std::string filterFile, std::string biasFile, Dims5 dims_in,
                       Dims3 padding, Dims3 stride, Dims3 dilation)
{
    // Assign all dimensional information
    this->dims_in = dims_in;
    this->pad = padding;
    this->str = stride;
    this->dil = dilation;

    // Obtain filter dimensions from file
    // TsrDims tsr_dims_filter = filename2dims(filterFile);
    this->dims_filter = filename2dims5(filterFile);

    // Load filter from file
    SizedArrayFloat filter_weights = readTensor2FloatBuffer(filterFile);
    SizedArrayFloat bias_values = readTensor2FloatBuffer(biasFile);

    // Create tensor, filter, and convolution descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnCreateFilterDescriptor(&desc_filter));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&desc_conv));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_bias));
    checkCUDNN(cudnnCreateActivationDescriptor(&desc_activation));

    // Set input tensor descriptor
    //      First, get memory access strides
    int data_in_stride[5] = {GET_DIM5_C(dims_in) * GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_D(dims_in) * GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in),
                             GET_DIM5_H(dims_in) * GET_DIM5_W(dims_in), GET_DIM5_W(dims_in), 1};
    //      Then, set the tensor descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_in,
        CUDNN_DATA_FLOAT,
        Conv3d_TENSOR_KERN_DIM,
        this->dims_in.dims,
        data_in_stride));

    // Set convolution descriptor
    checkCUDNN(cudnnSetConvolutionNdDescriptor(
        desc_conv,
        Conv3d_TENSOR_KERN_DIM - 2,
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
        Conv3d_TENSOR_KERN_DIM,
        this->dims_filter.dims));

    // Get output tensor dimensions
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
        desc_conv,
        desc_in,
        desc_filter,
        Conv3d_TENSOR_KERN_DIM,
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
        Conv3d_TENSOR_KERN_DIM,
        this->dims_out.dims,
        data_out_stride));

    // Set bias descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(
        desc_bias,
        CUDNN_DATA_FLOAT,
        Conv3d_TENSOR_KERN_DIM,
        this->dims_out.dims,
        data_out_stride));

    // Allocate space on GPU for filter and delete from host
    cudaMalloc(&dev_filter, filter_weights.count * sizeof(float));
    cudaMemcpy(dev_filter, filter_weights.arr, filter_weights.count * sizeof(float), cudaMemcpyHostToDevice);
    delete[] filter_weights.arr;

    // Allocate space on GPU for bias and delete from host
    cudaMalloc(&dev_bias, bias_values.count * sizeof(float));
    cudaMemcpy(dev_bias, bias_values.arr, bias_values.count * sizeof(float), cudaMemcpyHostToDevice);
    delete[] bias_values.arr;

    // Set activation descriptor (to IDENTITY)
    checkCUDNN(cudnnSetActivationDescriptor(
        desc_activation,
        CUDNN_ACTIVATION_IDENTITY,
        CUDNN_NOT_PROPAGATE_NAN,
        zero));
}

void Conv3dBias::run(cudnnHandle_t h, cudnnTensorDescriptor_t const *inputDesc, float *input,
                     cudnnTensorDescriptor_t *outputDesc, float **output, TagUnionExtraRet *extra)
{
    // Allocate space on GPU for output tensor
    int num_elements_out = dims_out.dims[0] * dims_out.dims[1] * dims_out.dims[2] * dims_out.dims[3] * dims_out.dims[4];
    cudaMalloc(output, num_elements_out * sizeof(float));
    cudaMemset(*output, 0, num_elements_out * sizeof(float));

    // Initialize the algorithm
    cudnnConvolutionFwdAlgoPerf_t algorithm_perf;
    int returned_algorithms = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        h,
        desc_in,
        desc_filter,
        desc_conv,
        desc_out,
        1,
        &returned_algorithms,
        &algorithm_perf));

    // Allocate workspace
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        h,
        desc_in,
        desc_filter,
        desc_conv,
        desc_out,
        algorithm_perf.algo,
        &dev_workspace_bytes));
    cudaMalloc(&dev_workspace, dev_workspace_bytes);

    // Run the convolution
    checkCUDNN(cudnnConvolutionBiasActivationForward(
        h,
        &one,
        desc_in,
        input,
        desc_filter,
        dev_filter,
        desc_conv,
        algorithm_perf.algo,
        dev_workspace,
        dev_workspace_bytes,
        &zero,
        desc_out,
        *output,
        desc_bias,
        dev_bias,
        desc_activation,
        desc_out,
        *output));

    return;
}

Dims5 Conv3dBias::getOutputDim()
{
    return dims_out;
}
