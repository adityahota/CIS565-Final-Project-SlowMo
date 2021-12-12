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
    lReluKern<<<numBlocks, blockSize>>>(numElem, coeff, (float *)input);
    cudaDeviceSynchronize();
}
