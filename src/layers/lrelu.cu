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
    // if (idx & ((1 << 8)))
    // {
    //     printf("%f, ", ptr[idx]);
    // }
    ptr[idx] = 1.f / (1.f + expf(-1.f * ptr[idx]));
    // if (idx & ((1 << 8)))
    // {
    //     printf("%f\n", ptr[idx]);
    // }
}

void LReLU::run(int n, float *input)
{
    // *output = input;
    // size_t numElem = 0;
    // checkCUDNN(cudnnGetTensorSizeInBytes(*inputDesc, &numElem));
    // numElem /= sizeof(float);
    int numBlocks = (n + blockSize - 1) / blockSize;
    lReluKern<<<numBlocks, blockSize>>>(n, coeff, input);
    cudaDeviceSynchronize();
}

void Sigmoid::run(int n, float *input)
{
    // printf("simoid run\n");
    // *output = input;
    // size_t numElem = 0;
    // checkCUDNN(cudnnGetTensorSizeInBytes(*inputDesc, &numElem));
    // // printf("tensor bytes : %i\n", numElem);
    // numElem /= sizeof(float);
    int numBlocks = (n + blockSize - 1) / blockSize;
    // printf("numBlocks is %i, blocksize is %i\n", numBlocks, blockSize);
    sigmoidKern<<<numBlocks, blockSize>>>(n, input);
    cudaDeviceSynchronize();
    // cudnnActivationDescriptor_t a;
    // checkCUDNN(cudnnCreateActivationDescriptor(&a));
    // checkCUDNN(cudnnSetActivationDescriptor(a, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.f));
    // checkCUDNN(cudnnActivationForward(h, a, &one, *inputDesc, input, &zero, *inputDesc, input));
    // checkCUDNN(cudnnDestroyActivationDescriptor(a));
}
