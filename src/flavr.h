#pragma once
#include "includes.h"

class Flavr
{
public:
    // Maybe take in vector of weights or something? to init the network?
    Flavr();
    // reads 4 input frames from pointer, how to ensure safety??, must copy inputs to gpu?
    // should it attach the frame before the interpolated ones? the one after? this will change main
    std::vector<VidFrame> runModel(VidFrame *input_frames);
};
