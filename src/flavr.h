#pragma once
#include "includes.h"
#include "layers/nnLeftBlock.h"
#include "layers/nnRightBlock.h"
#include "layers/nnCombineBlock.h"
#include "layers/nnPred.h"

class Flavr
{
public:
    // Maybe take in vector of weights or something? to init the network?
    Flavr();
    ~Flavr();
    // reads 4 input frames from pointer, how to ensure safety??, must copy inputs to gpu?
    // should it attach the frame before the interpolated ones? the one after? this will change main
    std::vector<VidFrame> runModel(VidFrame *input_frames);

private:
    // OrangeBlock l64;
    NNLeftBlock conv64;
    NNCombineBlock unet128;
    NNRightBlock tConv64;
    NNPred predLayer;
};
