#pragma once

// #include "includes.h"
// #include "layers/nnlayer.h"
// #include "layers/nnLeftBlock.h"
// #include "layers/nnRightBlock.h"
// #include "layers/nnCombineBlock.h"
// #include "layers/nnPred.h"

#include "layers/unmean.h"
#include "layers/unet.h"
#include "layers/postproc.h"
#include "layers/remean.h"

#include "layers/conv3d.h"

class Flavr
{
public:
    // Maybe take in vector of weights or something? to init the network?
    Flavr();
    ~Flavr();
    // reads 4 input frames from pointer, how to ensure safety??, must copy inputs to gpu?
    // should it attach the frame before the interpolated ones? the one after? this will change main
    //! Bad arguments; probably use cv::Mats or something
    std::vector<VidFrame> runModel(VidFrame *input_frames)
    {
        cudnnTensorDescriptor_t intoUNetDesc, outOfUNetDesc, afterTailDesc, outFlavrDesc;
        TagUnionExtraRet rgbTensorBundle;
        // TODO: Copy over data and process nchw
        unMean.run(h, nullptr, inFlavr, &intoUNetDesc, &intoUNet, &rgbTensorBundle);
        if (!addMean.updateMeans(&rgbTensorBundle))
        {
            //! throw error and dont continue
        }
        // uNet.run(h, intoUNet, outOfUNet);
        // tail.run(h, outOfUNet, afterTail);
        addMean.run(h, &afterTailDesc, afterTail, &outFlavrDesc, &outFlavr, nullptr);
        // TODO: Copy back data and process nchw
    }

private:
    cudnnHandle_t h;
    UnMean unMean;
    UNet uNet;
    // PostProc tail;
    ReMean addMean;
    float *inFlavr, *intoUNet, *outOfUNet, *afterTail, *outFlavr;
};

#if 0
class Conv3dTestWrapper
{
public:
    Conv3dTestWrapper() : spatioTemp(Conv3d(true))
    {
        // this->spatioTemp = Conv3d(true);
        checkCUDNN(cudnnCreate(&h)); // h done
        cudaMalloc(&dev_input, tensLen * sizeof(float));
        cudaMalloc(&dev_output, tensLen * sizeof(float));
        cudaMalloc(&dev_scratch, tensLen * sizeof(float));
        checkCUDAError("init conv3dTestWrapper");
    }
    ~Conv3dTestWrapper()
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        cudaFree(dev_scratch);
        checkCUDNN(cudnnDestroy(h));
        spatioTemp.~Conv3d();
    };
    cv::Mat run3dConv(cv::Mat nhwcIn)
    {
        auto imgSize = nhwcIn.total() * nhwcIn.elemSize();
        cudaMemcpy(dev_scratch, nhwcIn.data, imgSize, cudaMemcpyHostToDevice);
        // TODO convert format to NCDHW
        spatioTemp.run(h, nullptr, dev_input, nullptr, dev_output, nullptr);
        // TODO convert back to NHWC/NDHWC
        cudaMemcpy(nhwcIn.data, dev_scratch, imgSize, cudaMemcpyDeviceToHost);
        return nhwcIn; // may be not good pass by value?
    }

private:
    Conv3d spatioTemp;
    cudnnHandle_t h; //! NB only the outer layer should have a handle
    float *dev_input;
    float *dev_output;
    float *dev_scratch;

    float one = 1.f;
    float zero = 0.f;
    float negOne = -1.f;

    int nn = 1;
    int cc = 3;
    int dd = 1;
    int hh = 512;
    int ww = 512;
    int tensLen = nn * cc * dd * hh * ww;
};
#endif
