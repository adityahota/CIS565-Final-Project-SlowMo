#pragma once
#include "runnable.h"

class Conv3d : Runnable
{
public:
    ~Conv3d();
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
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

    Conv3d(std::string filterFile, std::string biasFile)
    {
        // Create Things
        // input descriptor (hardcoded)
        checkCUDNN(cudnnCreateTensorDescriptor(&inDescT));
        // filter descriptor (gotten from filename)
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        // dev_filter (read file and cudamalloc+cudamemcpy)
        SizedArrayFloat filtData = readTensor2FloatBuffer(filterFile);
        cudaMalloc(&dev_filter, filtData.count * sizeof(float));
        cudaMemcpy(dev_filter, filtData.arr, filtData.count * sizeof(float), cudaMemcpyHostToDevice);
        delete[] filtData.arr;
        // convDesc ??
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        // biasDesc (gotten from filename)
        checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc));
        // dev_bias (read file and cudamalloc+cudamemcpy)
        SizedArrayFloat biasData = readTensor2FloatBuffer(biasFile);
        cudaMalloc(&dev_bias, biasData.count * sizeof(float));
        cudaMemcpy(dev_bias, biasData.arr, biasData.count * sizeof(float), cudaMemcpyHostToDevice);
        delete[] biasData.arr;
        // activDesc ??
        checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
        // outDescT ??
        checkCUDNN(cudnnCreateTensorDescriptor(&outDescT));
        // algo ??
        // workspaceSize (use cudnn to calc)
        // dev_workspace (cudamalloc + cudamemcpy)
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

//*************************************************************************************************

/*
checkCUDNN(cudnnSetTensorNdDescriptor(desc_in,
                                  CUDNN_DATA_FLOAT,
                                  v_dim_in.size(),
                                  v_dim_in.data(),
                                  v_str_in.data()));
*/
