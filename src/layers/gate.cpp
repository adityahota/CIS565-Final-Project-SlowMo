#include "gate.h"

Gate::Gate(Dims5 poolDimsIn, std::string filterFile, std::string biasFile, cudnnHandle_t h)
    : gateInputDims(poolDimsIn)
{
    // Adaptive Average Pool 3D has output dimension 1, meaning window size is same as input size DHW
    Dims3 pool1_win = mkDims3(GET_DIM5_D(poolDimsIn), GET_DIM5_H(poolDimsIn), GET_DIM5_W(poolDimsIn));
    Dims3 pool1_str = pool1_win;
    Dims3 pool1_pad = mkDims3(0, 0, 0);
    pool = new AvgPool3d(poolDimsIn, pool1_win, pool1_pad, pool1_str);

    // Conv3d layer
    Dims5 fcDimsIn = pool->getOutputDim();
    Dims3 fcPadding = zeroDims3; // Always 0
    Dims3 fcStride = unitDims3;  // Always 1
    Dims3 fcDilate = unitDims3;  // Always 1
    fcLayer = new Conv3dBias(filterFile, biasFile, fcDimsIn, fcPadding, fcStride, fcDilate, h);

    // Sigmoid layer
    Dims5 tmp = fcLayer->getOutputDim();
    sigmoidSize = dims5ToSize(tmp);
    s = new Sigmoid();

    checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc));
    checkCUDNN(cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

    cudnnCreateTensorDescriptor(&postFCDesc);
    cudnnSetTensorNdDescriptor(postFCDesc, CUDNN_DATA_FLOAT, 5, tmp.dims, dim5Stride(tmp).dims);
    cudnnCreateTensorDescriptor(&multiplyDesc);
    cudnnSetTensorNdDescriptor(multiplyDesc, CUDNN_DATA_FLOAT, 5, gateInputDims.dims, dim5Stride(gateInputDims).dims);
    cudaMalloc(&dev_outBuf, dims5ToSize(gateInputDims) * sizeof(float));
}

void Gate::run(cudnnHandle_t h,
               cudnnTensorDescriptor_t const *inputDesc, float *input,
               cudnnTensorDescriptor_t *outputDesc, float **output,
               TagUnionExtraRet *extra)
{
    // Run pooling and store output in postPool (allocated by pooling layer)
    pool->run(h, nullptr, input, nullptr, &postPool, nullptr);
    // checkCUDAError("cuda gate postrun");

    // Run the FC layer (Conv3d) and store output in postFC (allocated by Conv3d layer)
    fcLayer->run(h, nullptr, postPool, nullptr, &postFC, nullptr);
    // checkCUDAError("cuda gate post fc");

    // Run the Sigmoid layer (data is mutated in place)
    s->run(sigmoidSize, postFC);
    // checkCUDAError("cuda gate post sigmoid");

    // C = op( alpha1 * A, alpha2 * B ) + beta * C
    checkCUDNN(cudnnOpTensor(h,
                             opDesc,
                             &one, multiplyDesc, input,
                             &one, postFCDesc, postFC,
                             &zero, multiplyDesc, dev_outBuf)); // Multiply cudnnOpTensor() output = input * output of sigmoid
    // checkCUDAError("cuda gate post multiply");

    *output = dev_outBuf;
}

Gate::~Gate()
{
    cudaFree(postFC);
    cudaFree(postPool);
    // fcLayer->~Conv3dBias();
    // pool->~AvgPool3d();
    delete pool;
    delete fcLayer;
    delete s;
    checkCUDNN(cudnnDestroyTensorDescriptor(postFCDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(multiplyDesc));
    checkCUDNN(cudnnDestroyOpTensorDescriptor(opDesc));
}

Dims5 Gate::getOutputDims()
{
    return gateInputDims;
}
