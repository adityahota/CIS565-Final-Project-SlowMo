#include "lrelu.h"

__global__ void lReluKern(int n, float coeff, float *ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    ptr[idx] *= ptr[idx] > 0.f ? 1.f : coeff;
}
__global__ void sigmoidKern(int n, float *ptr)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    ptr[idx] = 1.f / (1.f + expf(-1.f * ptr[idx]));
}

void LReLU::run(cudnnHandle_t h,
                cudnnTensorDescriptor_t const *inputDesc, float *input,
                cudnnTensorDescriptor_t *outputDesc, float **output,
                TagUnionExtraRet *extra)
{
    *output = input;
    size_t numElem = 0;
    checkCUDNN(cudnnGetTensorSizeInBytes(*inputDesc, &numElem));
    numElem /= sizeof(float);
    int numBlocks = (numElem + blockSize - 1) / blockSize;
    lReluKern<<<numBlocks, blockSize>>>(numElem, coeff, input);
    cudaDeviceSynchronize();
}

void Sigmoid::run(cudnnHandle_t h,
                  cudnnTensorDescriptor_t const *inputDesc, float *input,
                  cudnnTensorDescriptor_t *outputDesc, float **output,
                  TagUnionExtraRet *extra)
{
    *output = input;
    size_t numElem = 0;
    checkCUDNN(cudnnGetTensorSizeInBytes(*inputDesc, &numElem));
    numElem /= sizeof(float);
    int numBlocks = (numElem + blockSize - 1) / blockSize;
    sigmoidKern<<<numBlocks, blockSize>>>(numElem, input);
    cudaDeviceSynchronize();
}
