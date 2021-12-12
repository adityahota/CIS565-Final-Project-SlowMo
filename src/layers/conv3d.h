#pragma once
#include "runnable.h"

#define Conv3d_TENSOR_KERN_DIM 5
#define CONV2D_TENSOR_KERN_DIM 4

class Conv3dBias : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;

    Conv3dBias(std::string filterFile, std::string biasFile, Dims5 dims_in,
               Dims3 padding, Dims3 stride, Dims3 dilation);

    ~Conv3dBias()
    {
        cudnnDestroyTensorDescriptor(desc_in);
        cudnnDestroyTensorDescriptor(desc_out);
        cudnnDestroyFilterDescriptor(desc_filter);
        cudnnDestroyConvolutionDescriptor(desc_conv);
        cudnnDestroyTensorDescriptor(desc_bias);
        cudnnDestroyActivationDescriptor(desc_activation);

        cudaFree(dev_filter);
        cudaFree(dev_workspace);
        cudaFree(dev_bias);
    }

    Dims5 getOutputDim();

private:
    // Descriptors
    cudnnTensorDescriptor_t desc_in;
    cudnnTensorDescriptor_t desc_out;
    cudnnFilterDescriptor_t desc_filter;
    cudnnConvolutionDescriptor_t desc_conv;
    cudnnTensorDescriptor_t desc_bias;
    cudnnActivationDescriptor_t desc_activation;

    // Tensor dimensions
    Dims5 dims_in;
    Dims5 dims_out;
    Dims5 dims_filter;

    // Convolution parameters (padding, stride, dilation)
    Dims3 pad;
    Dims3 str;
    Dims3 dil;

    // Algorithm for convolution
    cudnnConvolutionFwdAlgo_t algo;

    // GPU data
    float *dev_filter;
    float *dev_workspace;
    float *dev_bias;

    // Other information
    size_t dev_workspace_bytes;
};

class Conv3d : Runnable
{
public:
    /**
     * @brief
     *
     * @param h
     * @param inputDesc
     * @param input
     * @param outputDesc
     * @param output Passed in; self malloc's; caller responsible for freeing
     * @param extra
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;

    Conv3d(std::string filterFile, Dims5 dims_in,
           Dims3 padding, Dims3 stride, Dims3 dilation);

    ~Conv3d()
    {
        cudnnDestroyTensorDescriptor(desc_in);
        cudnnDestroyTensorDescriptor(desc_out);
        cudnnDestroyFilterDescriptor(desc_filter);
        cudnnDestroyConvolutionDescriptor(desc_conv);

        cudaFree(dev_filter);
        cudaFree(dev_workspace);
    }

    Dims5 getOutputDim();

private:
    // Descriptors
    cudnnTensorDescriptor_t desc_in;
    cudnnTensorDescriptor_t desc_out;
    cudnnFilterDescriptor_t desc_filter;
    cudnnConvolutionDescriptor_t desc_conv;

    // Tensor dimensions
    Dims5 dims_in;
    Dims5 dims_out;
    Dims5 dims_filter;

    // Convolution parameters (padding, stride, dilation)
    Dims3 pad;
    Dims3 str;
    Dims3 dil;

    // Algorithm for convolution
    cudnnConvolutionFwdAlgo_t algo;

    // GPU data
    float *dev_filter;
    float *dev_workspace;

    // Other information
    size_t dev_workspace_bytes;
};

//*************************************************************************************************

/*
checkCUDNN(cudnnSetTensorNdDescriptor(desc_in,
                                  CUDNN_DATA_FLOAT,
                                  v_dim_in.size(),
                                  v_dim_in.data(),
                                  v_str_in.data()));
*/

class Conv2dBias : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        // y = Act( alpha1 * conv(x) + alpha2 * z + bias )
        // z and y can alias to same buffer but x cannot overlap them
        checkCUDNN(cudnnConvolutionBiasActivationForward(
            h,
            &one,
            inDescT, input,               // conv input
            filterDesc, dev_filter,       // conv filter
            convDesc, algo,               // conv algo
            dev_workspace, workspaceSize, // conv workspace
            &zero, outDescT, output,      // z
            biasDesc, dev_bias,           // bias
            activDesc,                    // activation
            outDescT, output              // output
            ));
    }

    // TODO: Handle case where no bias is given by using properly sized tensor of 0.f
    Conv2dBias(std::string filterFile, std::string biasFile, cudnnActivationMode_t actMode)
    {
        // Create Things

        // input descriptor (hardcoded)
        checkCUDNN(cudnnCreateTensorDescriptor(&inDescT));
        // checkCUDNN(cudnnSetTensorNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // filter descriptor (gotten from filename)
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        // checkCUDNN(cudnnSetFilterNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // dev_filter (read file and cudamalloc+cudamemcpy)
        SizedArrayFloat filtData = readTensor2FloatBuffer(filterFile);
        cudaMalloc(&dev_filter, filtData.count * sizeof(float));
        cudaMemcpy(dev_filter, filtData.arr, filtData.count * sizeof(float), cudaMemcpyHostToDevice);
        delete[] filtData.arr;

        // convDesc ??
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

        // biasDesc (gotten from filename)
        checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
        // checkCUDNN(cudnnSetTensorNdDescriptor(biasDesc, CUDNN_DATA_FLOAT, ));

        // dev_bias (read file and cudamalloc+cudamemcpy)
        SizedArrayFloat biasData = readTensor2FloatBuffer(biasFile);
        cudaMalloc(&dev_bias, biasData.count * sizeof(float));
        cudaMemcpy(dev_bias, biasData.arr, biasData.count * sizeof(float), cudaMemcpyHostToDevice);
        delete[] biasData.arr;

        // activDesc ??
        checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN(cudnnSetActivationDescriptor(activDesc, actMode, CUDNN_NOT_PROPAGATE_NAN, MAXFLOAT));

        // outDescT ??
        checkCUDNN(cudnnCreateTensorDescriptor(&outDescT));
        // checkCUDNN(cudnnSetTensorNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // algo ??
        // workspaceSize (use cudnn to calc)
        // dev_workspace (cudamalloc + cudamemcpy)
    }

    ~Conv2dBias()
    {
        cudnnDestroyTensorDescriptor(inDescT);
        cudnnDestroyTensorDescriptor(outDescT);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyActivationDescriptor(activDesc);
        cudaFree(dev_filter);
        cudaFree(dev_workspace);

        cudnnDestroyTensorDescriptor(biasDesc);
        cudaFree(dev_bias);
    }

private:
    cudnnTensorDescriptor_t inDescT;

    cudnnFilterDescriptor_t filterDesc;
    float *dev_filter;

    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;

    size_t workspaceSize;
    float *dev_workspace;

    cudnnTensorDescriptor_t biasDesc;
    float *dev_bias;

    cudnnActivationDescriptor_t activDesc;
    cudnnTensorDescriptor_t outDescT;
};

class Conv2d : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {

        // checkCUDNN(cudnnConvolutionForward());
        // activation
    }

    // TODO: Handle case where no bias is given by using properly sized tensor of 0.f
    Conv2d(std::string filterFile, std::string biasFile, cudnnActivationMode_t actMode)
    {
        // Create Things

        // input descriptor (hardcoded)
        checkCUDNN(cudnnCreateTensorDescriptor(&inDescT));
        // checkCUDNN(cudnnSetTensorNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // filter descriptor (gotten from filename)
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        // checkCUDNN(cudnnSetFilterNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // dev_filter (read file and cudamalloc+cudamemcpy)
        SizedArrayFloat filtData = readTensor2FloatBuffer(filterFile);
        cudaMalloc(&dev_filter, filtData.count * sizeof(float));
        cudaMemcpy(dev_filter, filtData.arr, filtData.count * sizeof(float), cudaMemcpyHostToDevice);
        delete[] filtData.arr;

        // convDesc ??
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

        // activDesc ??
        checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN(cudnnSetActivationDescriptor(activDesc, actMode, CUDNN_NOT_PROPAGATE_NAN, MAXFLOAT));

        // outDescT ??
        checkCUDNN(cudnnCreateTensorDescriptor(&outDescT));
        // checkCUDNN(cudnnSetTensorNdDescriptor(inDescT, CUDNN_DATA_FLOAT, ));

        // algo ??
        // workspaceSize (use cudnn to calc)
        // dev_workspace (cudamalloc + cudamemcpy)
    }

    ~Conv2d()
    {
        cudnnDestroyTensorDescriptor(inDescT);
        cudnnDestroyTensorDescriptor(outDescT);
        cudnnDestroyFilterDescriptor(filterDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnnDestroyActivationDescriptor(activDesc);
        cudaFree(dev_filter);
        cudaFree(dev_workspace);
    }

private:
    cudnnTensorDescriptor_t inDescT;

    cudnnFilterDescriptor_t filterDesc;
    float *dev_filter;

    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;

    size_t workspaceSize;
    float *dev_workspace;

    cudnnActivationDescriptor_t activDesc;
    cudnnTensorDescriptor_t outDescT;
};
