#pragma once

#include "includes.h"
#include "layers/nnlayer.h"
// #include "layers/nnLeftBlock.h"
// #include "layers/nnRightBlock.h"
// #include "layers/nnCombineBlock.h"
// #include "layers/nnPred.h"
#if 1
class Runnable
{
public:
    virtual void run(cudnnHandle_t h, void *input, void *output) = 0;
};
class UnMean : Runnable
{
public:
    void run(cudnnHandle_t h, void *input, void *output) override
    {
        checkCUDNN(
            cudnnReduceTensor(h,
                              tensReduceDescT,
                              NULL, 0,
                              wkspc, wkspcSize, // todo
                              &one, inDescT, input,
                              &zero, rgbAvgsDescT, rgbAvgs));
        checkCUDNN(
            cudnnOpTensor(h,
                          tensOpDescT,
                          &one, inDescT, input,
                          &negOne, rgbAvgsDescT, rgbAvgs,
                          &zero, outDescT, output));
    }

private:
    cudnnReduceTensorDescriptor_t tensReduceDescT; // CUDNN_REDUCE_TENSOR_AVG, dim 1x3x1x1x1
    void *wkspc;
    size_t wkspcSize;
    float one = 1.f;
    float zero = 0.f;
    cudnnTensorDescriptor_t inDescT;
    void *rgbAvgs;
    cudnnTensorDescriptor_t rgbAvgsDescT;
    cudnnOpTensorDescriptor_t tensOpDescT;
    float negOne = -1.f;
    cudnnTensorDescriptor_t outDescT;
};
class UNet : Runnable
{
};
class PostProc : Runnable
{
};
class ReMean : Runnable
{
};
#endif

class Flavr
{
public:
    // Maybe take in vector of weights or something? to init the network?
    Flavr();
    ~Flavr();
    // reads 4 input frames from pointer, how to ensure safety??, must copy inputs to gpu?
    // should it attach the frame before the interpolated ones? the one after? this will change main
    // std::vector<VidFrame> runModel(VidFrame *input_frames);
    std::vector<VidFrame> runModel(VidFrame *input_frames)
    {
        // TODO: Copy over data and process nchw
        unmean.run(h, inFlavr, intoUNet);
        uNet.run(h, intoUNet, outOfUNet);
        tail.run(h, outOfUNet, afterTail);
        addMean.run(h, afterTail, outFlavr);
        // TODO: Copy back data and process nchw
    }

private:
    cudnnHandle_t h;
    UnMean unmean;
    UNet uNet;
    PostProc tail;
    ReMean addMean;
    void *inFlavr, *intoUNet, *outOfUNet, *afterTail, *outFlavr;
};
