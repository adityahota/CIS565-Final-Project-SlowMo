#pragma once

#include "includes.h"
#include "layers/nnlayer.h"
// #include "layers/nnLeftBlock.h"
// #include "layers/nnRightBlock.h"
// #include "layers/nnCombineBlock.h"
// #include "layers/nnPred.h"

// __global__ void NHWC_2_NCHW(int n, float *in, float *out, int batch, int h, int w, int c)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n)
//     {
//         return;
//     }
//     auto in_c = idx;
//     auto in_n = in_c / (h * w * c);
//     in_c -= in_n * h * w * c;
//     auto in_h = in_c / (w * c);
//     in_c -= in_h * w * c;
//     auto in_w = in_c / c;
//     in_c -= in_w * c;
//     auto outdex = in_w + w * in_h + w * h * in_c + w * h * c * in_n;
// }
// __global__ void NCHW_2_NHWC(int n, float *in, float *out, int batch, int h, int w, int c)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n)
//     {
//         return;
//     }
//     // TODO
//     //  auto in_c = idx;
//     //  auto in_n = in_c / (h * w * c);
//     //  in_c -= in_n * h * w * c;
//     //  auto in_h = in_c / (w * c);
//     //  in_c -= in_h * w * c;
//     //  auto in_w = in_c / c;
//     //  in_c -= in_w * c;
//     //  auto outdex = in_w + w * in_h + w * h * in_c + w * h * c * in_n;
// }
// __global__ void NHWC_2_NCDHW(int n, float *in, float *out, int batch, int c);
#if 1
typedef struct TensDescAndData
{
    cudnnTensorDescriptor_t desc;
    void *tens;
} TensDescAndData;
enum ExtraReturnTag
{
    NIL,
    TENSOR_DATA,
};
typedef union ExtraRetVal
{
    void *nothing;
    TensDescAndData tensorBundle;
} ExtraRetVal;

/**
 * @brief Nothing | (TensorDescriptor, StartOfTensor) [Feel free to add more to suit needs]
 */
typedef struct TagUnionExtraRet
{
    ExtraReturnTag tag;
    ExtraRetVal val;
} TagUnionExtraRet;

class Runnable
{
public:
    /**
     * @brief Runs the underlying NN Layers
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input,
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output !!Should run make this or should it be premade?
     * @param output start of the ouput tensor data
     * @param extra if any extra information needs to be passed along
     */
    virtual void run(cudnnHandle_t h,
                     cudnnTensorDescriptor_t const *inputDesc, void *input,
                     cudnnTensorDescriptor_t *outputDesc, void *output,
                     TagUnionExtraRet *extra) = 0;

    float one = 1.f;
    float zero = 0.f;
    float negOne = -1.f;
};
class UnMean : Runnable
{
public:
    /**
     * @brief Finds the RGB Means of the input frames and normalizes them
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output
     * @param output start of the ouput tensor data
     * @param extra passes out rgb averages for the ReMean step
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        checkCUDNN(cudnnReduceTensor(h, tensReduceDescT,
                                     NULL, 0,
                                     wkspc, wkspcSize, // todo
                                     &one, inDescT, input,
                                     &zero, rgbAvgsDescT, rgbAvgs));
        checkCUDNN(cudnnAddTensor(h,
                                  &negOne, rgbAvgsDescT, rgbAvgs,
                                  &one, inDescT, input));
        // Assuming everything is pass by value'ed, since extras is a class field, it should live long enough
        {
            auto rgbBundle = TensDescAndData{rgbAvgsDescT, rgbAvgs};
            ExtraRetVal onion;
            onion.tensorBundle = rgbBundle;
            extras.tag = TENSOR_DATA;
            extras.val = onion; // maybe errors with stack, moving, ownership, lifetimes
            *extra = extras;
        }
        *outputDesc = inDescT;
        output = input;
    }

private:
    cudnnReduceTensorDescriptor_t tensReduceDescT; // CUDNN_REDUCE_TENSOR_AVG
    void *wkspc;
    size_t wkspcSize;
    cudnnTensorDescriptor_t inDescT; // dim 1x3x4xHxW
    void *rgbAvgs;
    cudnnTensorDescriptor_t rgbAvgsDescT;  // dim 1x3x1x1x1
    cudnnOpTensorDescriptor_t tensOpDescT; // CUDNN_OP_TENSOR_ADD
    cudnnTensorDescriptor_t outDescT;      // dim 1x3x4xHxW
    TagUnionExtraRet extras;
};
// class UNet : Runnable
// {
// };
// class PostProc : Runnable
// {
// };
class ReMean : Runnable
{
public:
    /**
     * @brief Adds back the rgb means to the colors
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output
     * @param output start of the ouput tensor data
     * @param extra unused
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        checkCUDNN(cudnnAddTensor(h,
                                  &negOne, avgTsrDesc, avgTsr,
                                  &one, *inputDesc, input));
    }
    bool updateMeans(TagUnionExtraRet *extra)
    {
        if (extra->tag == TENSOR_DATA)
        {
            avgTsrDesc = extra->val.tensorBundle.desc;
            avgTsr = extra->val.tensorBundle.tens;
            return true;
        }
        return false;
    }

private:
    cudnnTensorDescriptor_t avgTsrDesc;
    void *avgTsr;
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
    std::vector<VidFrame> runModel(VidFrame *input_frames)
    {
        cudnnTensorDescriptor_t intoUNetDesc, outOfUNetDesc, afterTailDesc, outFlavrDesc;
        TagUnionExtraRet rgbTensorBundle;
        // TODO: Copy over data and process nchw
        unMean.run(h, nullptr, inFlavr, &intoUNetDesc, intoUNet, &rgbTensorBundle);
        if (!addMean.updateMeans(&rgbTensorBundle))
        {
            //! throw error and dont continue
        }
        // uNet.run(h, intoUNet, outOfUNet);
        // tail.run(h, outOfUNet, afterTail);
        addMean.run(h, &afterTailDesc, afterTail, &outFlavrDesc, outFlavr, nullptr);
        // TODO: Copy back data and process nchw
    }

private:
    cudnnHandle_t h;
    UnMean unMean;
    // UNet uNet;
    // PostProc tail;
    ReMean addMean;
    void *inFlavr, *intoUNet, *outOfUNet, *afterTail, *outFlavr;
};

class Conv3d : Runnable
{
public:
    Conv3d() {}
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

class Conv3dTestWrapper
{
public:
    Conv3dTestWrapper()
    {
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
