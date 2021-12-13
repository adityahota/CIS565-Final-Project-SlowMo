#pragma once

#include "conv3d.h"
#include "lrelu.h"

/**
 * @brief Class that is the Basic Stem: conv relu; only used once
 *
 */
class BStem : Runnable
{
public:
    /**
     * @brief Concrete run implementation
     *
     * @param h handle
     * @param inputDesc unused
     * @param input input tensor; passed in; self does not own
     * @param outputDesc unused
     * @param output output tensor passed in, malloc'ed by the conv layer, caller responsible for freeing
     * @param extra unused
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        //  l.run() no bias, activation relu
        float boo = 0;
        float *boo_loc = &boo;
        float **tmp = &boo_loc;
        l->run(h, &inDescT, input, &outDescT, output, nullptr);
        relu->run(reluSize, *output);
        //? does this set output correctly?
    }

    BStem(Dims5 blockDimsIn, std::string stem_weights)
    {
        Dims3 paddingDims = mkDims3(1, 3, 3);
        Dims3 strideDims = mkDims3(1, 2, 2);
        Dims3 dilationDims = unitDims3;

        l = new Conv3d(stem_weights, blockDimsIn, paddingDims, strideDims, dilationDims);

        relu = new LReLU();
        reluSize = dims5ToSize(l->getOutputDim());

        outputDims = l->getOutputDim();
    }

    ~BStem()
    {
        l->~Conv3d();
        delete l;
        delete relu;
    }

    Dims5 getOutputDims()
    {
        return outputDims;
    }

private:
    Conv3d *l;
    LReLU *relu;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
    int reluSize;
    Dims5 outputDims;
};
