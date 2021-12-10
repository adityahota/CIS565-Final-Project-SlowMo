#pragma once

#include "nnlayer.h"

class NNLeakyRelu : NNLayer
{
private:
    int num_elements;

    float slope;
    float *data_input;
    float *data_output;

public:
    NNLeakyRelu(float negative_slope);

    void run(cudnnHandle_t cudnn_handle);

    void setData(int num_elements, float *input, float *output);
};
