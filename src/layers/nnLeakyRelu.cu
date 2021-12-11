#include "nnLeakyRelu.h"

__global__ void kernLeakyRelu(int n, float negative_slope,
        float *data_in, float *data_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    float value = data_in[idx];
    data_out[idx] = value > 0.f ? value : negative_slope * value;
}

NNLeakyRelu::NNLeakyRelu(float negative_slope)
  :  num_elements(0), slope(negative_slope),
     data_input(nullptr), data_output(nullptr)
{ }

void NNLeakyRelu::run(cudnnHandle_t cudnn_handle)
{
    dim3 fullBlocksPerGrid = ((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernLeakyRelu<<<fullBlocksPerGrid, BLOCK_SIZE>>>(num_elements, slope,
            data_input, data_output);
}

void NNLeakyRelu::setData(int num_elements, float *input, float *output)
{
    this->num_elements = num_elements;
    data_input = input;
    data_output = output;
}
