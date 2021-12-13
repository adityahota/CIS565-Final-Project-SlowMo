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
