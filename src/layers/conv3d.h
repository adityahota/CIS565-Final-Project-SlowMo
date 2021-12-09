#pragma once
#include "runnable.h"

class Conv3d : Runnable
{
public:
    Conv3d(std::string kernelPath)
    {
        // Read tensor file into the filter array

        float *buf;
        int len;
        std::ifstream is;
        is.open(kernelPath, std::ios::binary);
        is.seekg(0, std::ios::end);
        len = is.tellg();
        is.seekg(0, std::ios::beg);
        int numFloats = len / sizeof(float);
        buf = new float[numFloats];
        is.read((char *)buf, len);
        cudaMalloc(&dev_filter, len);
        cudaMemcpy(dev_filter, buf, len, cudaMemcpyHostToDevice);
        delete buf;
    }
    Conv3d(bool simpleTestCase);
    ~Conv3d();
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        checkCUDNN(cudnnConvolutionForward(h,
                                           &one, inDescT, input,
                                           filterDesc, dev_filter, convDesc,
                                           algo, workspace, workspaceSize,
                                           &zero, outDescT, output));
    }

private:
    float *dev_filter;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
    cudnnConvolutionFwdAlgo_t algo;
    void *workspace;
    size_t workspaceSize;
};
