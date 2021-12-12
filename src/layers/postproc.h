#pragma once
#if 0

// #include "runnable.h"

const int h_flattenTensor = 256; // todo
const int w_flattenTensor = 448; // todo

// operates on the raw tensor data; be sure to allocate nonoverllaping for the output
__global__ void flattenTensor(float *input,
                              float *output, int h_flattenTensor, int w_flattenTensor)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    if (c >= 64 * 4 || h >= h_flattenTensor || w >= w_flattenTensor)
    {
        return;
    }
    int o_idx = w +
                h * w_flattenTensor +
                c * h_flattenTensor * w_flattenTensor;
    int i_idx = w +
                h * w_flattenTensor +
                (c / 64) * h_flattenTensor * w_flattenTensor +
                (c - ((c / 64) * c) + (c / 64) * 4 * h_flattenTensor * w_flattenTensor);
    // output[0][c][h][w] = input[0][c - (c / 64) * c][c / 64][h][w];
    output[o_idx] = input[i_idx];
}

// Probably leave this for last
// flatten: [1][64][4][h][w]->[1][64*4][h][w]-> (NCDHW to NCHW)
// featurefuse:2dconv 64*4->64->lrelu->
// outconv: reflectionpad;
// outconv: conv2d 64->3*D->
// process into frames NCHW -> split c into chunks of 3 so rgb frames. now have N3HW (only do 2x)
class PostProc : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override
    {
        flattenTensor<<<numBlocks, threadsPerBlock>>>((float *)input, flattenScratch, h_flattenTensor, w_flattenTensor);
        featureFuse.run(h, &flattenOutDesc, flattenScratch, &ffOutDesc, &preRefPad, nullptr);
        lr.run(h, &ffOutDesc, preRefPad, nullptr, nullptr, nullptr); // todo
        //! reflection pad todo
        // outConv.run();
    }

private:
    Conv2dBias outConv;
    Conv2d featureFuse;
    LReLU lr;
    cudnnTensorDescriptor_t flattenOutDesc; // make this separately to apply post flattening
    dim3 numBlocks;
    dim3 threadsPerBlock;
    float *inputFlatten;
    float *flattenScratch;
    cudnnTensorDescriptor_t ffOutDesc;
    float *preRefPad;
};

#endif

