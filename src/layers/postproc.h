#pragma once

#include "runnable.h"

// Probably leave this for last
// flatten: [1][64][4][h][w]->[1][64*4][h][w]->
// featurefuse:2dconv 64*4->64->lrelu->
// outconv:reflectionpad; conv2d 64->3*D->
// process into frames NCHW -> split c into chunks of 3 so rgb frames. now have List of N3HW
class PostProc : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override;

private:
    Conv3d<true> outConv;
};
