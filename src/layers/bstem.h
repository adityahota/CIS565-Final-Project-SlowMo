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
        relu->run(h, &outDescT, *output, nullptr, tmp, nullptr);
        //? does this set output correctly?
    }

    BStem()
    {
        int tmp1[5] = {1, 3, 4, 256, 448};
        int tmp2[5] = {64, 3, 3, 7, 7};
        int tmp3[3] = {1, 3, 3};
        int tmp4[3] = {1, 2, 2};
        int tmp5[3] = {1, 1, 1};
        Dims5 inputDims = mkDims5(tmp1);
        Dims5 filterDims = mkDims5(tmp2);
        Dims3 paddingDims = mkDims3(tmp3);
        Dims3 strideDims = mkDims3(tmp4);
        Dims3 dilationDims = mkDims3(tmp5);
        l = new Conv3d("module.encoder.stem.0.weight__64x3x3x7x7.bin",
                       inputDims, paddingDims, strideDims, dilationDims);
        relu = new LReLU();
    }

    ~BStem()
    {
        l->~Conv3d();
        delete l;
        delete relu;
    }

private:
    Conv3d *l;
    LReLU *relu;
    cudnnTensorDescriptor_t inDescT;
    cudnnTensorDescriptor_t outDescT;
};
