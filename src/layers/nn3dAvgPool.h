#pragma once

#include "nnlayer.h"

#define NUM_AVG_POOL_DIM 3
#define NUM_AVG_POOL_TENSOR_DIM (NUM_AVG_POOL_DIM + 2)

class NN3dAvgPool: public NNLayer {
private:
    std::vector<int> dim_in = std::vector<int>(NUM_AVG_POOL_TENSOR_DIM);
    std::vector<int> dim_out = std::vector<int>(NUM_AVG_POOL_TENSOR_DIM);

    std::vector<int> size_win = std::vector<int>(NUM_AVG_POOL_DIM);
    std::vector<int> size_pad = std::vector<int>(NUM_AVG_POOL_DIM);
    std::vector<int> size_str = std::vector<int>(NUM_AVG_POOL_DIM);

    float *data_input;
    float *data_output;

    cudnnTensorDescriptor_t desc_in;
    cudnnTensorDescriptor_t desc_out;
    cudnnPoolingDescriptor_t desc_pool;

public:
    NN3dAvgPool(int N_in, int C_in, int D_in, int H_in, int W_in,
                int win_D, int win_H, int win_W,
                int p_D, int p_H, int p_W,
                int s_D, int s_H, int s_W);

    void run(cudnnHandle_t cudnn_handle);

    void setData(float *input, float *output);

    int getOutputN();
    int getOutputC();
    int getOutputD();
    int getOutputH();
    int getOutputW();
    int getOutputSize();
};

