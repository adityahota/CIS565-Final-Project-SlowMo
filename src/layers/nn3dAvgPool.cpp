#include "nn3dAvgPool.h"

NN3dAvgPool::NN3dAvgPool(int N_in, int C_in, int D_in, int H_in, int W_in,
                         int win_D, int win_H, int win_W,
                         int p_D, int p_H, int p_W,
                         int s_D, int s_H, int s_W)
{
    // Set data pointers to null
    data_input = nullptr;
    data_output = nullptr;

    // Set input tensor dimensions
    dim_in[0] = N_in;
    dim_in[1] = C_in;
    dim_in[2] = D_in;
    dim_in[3] = H_in;
    dim_in[4] = W_in;

    // Set window size, padding, and stride
    size_win[0] = win_D;
    size_win[1] = win_H;
    size_win[2] = win_W;

    size_pad[0] = p_D;
    size_pad[1] = p_H;
    size_pad[2] = p_W;

    size_str[0] = s_D;
    size_str[1] = s_H;
    size_str[2] = s_W;

    // Create tensor and pooling descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnCreatePoolingDescriptor(&desc_pool));

    // Set input tensor descriptor
    std::vector<int> v_str_in = {GET_DIM_C(dim_in) * GET_DIM_D(dim_in) * GET_DIM_H(dim_in) * GET_DIM_W(dim_in),
                                 GET_DIM_D(dim_in) * GET_DIM_H(dim_in) * GET_DIM_W(dim_in),
                                 GET_DIM_H(dim_in) * GET_DIM_W(dim_in), GET_DIM_W(dim_in), 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(desc_in,
                                          CUDNN_DATA_FLOAT,
                                          dim_in.size(),
                                          dim_in.data(),
                                          v_str_in.data()));
    // Set pooling descriptor
    checkCUDNN(cudnnSetPoolingNdDescriptor(desc_pool,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           NUM_AVG_POOL_DIM,
                                           size_win.data(),
                                           size_pad.data(),
                                           size_str.data()));

    // Get output tensor dimensions
    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(desc_pool,
                                                 desc_in,
                                                 dim_in.size(),
                                                 dim_out.data()))

    // Set output tensor descriptor
    std::vector<int> v_str_out = {GET_DIM_C(dim_out) * GET_DIM_D(dim_out) * GET_DIM_H(dim_out) * GET_DIM_W(dim_out),
                                  GET_DIM_D(dim_out) * GET_DIM_H(dim_out) * GET_DIM_W(dim_out),
                                  GET_DIM_H(dim_out) * GET_DIM_W(dim_out), GET_DIM_W(dim_out), 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(desc_out,
                                          CUDNN_DATA_FLOAT,
                                          dim_out.size(),
                                          dim_out.data(),
                                          v_str_out.data()));
}

void NN3dAvgPool::setData(float *input, float *output)
{
    data_input = input;
    data_output = output;
}

int NN3dAvgPool::getOutputN()
{
    return GET_DIM_N(dim_out);
}

int NN3dAvgPool::getOutputC()
{
    return GET_DIM_C(dim_out);
}

int NN3dAvgPool::getOutputD()
{
    return GET_DIM_D(dim_out);
}

int NN3dAvgPool::getOutputH()
{
    return GET_DIM_H(dim_out);
}

int NN3dAvgPool::getOutputW()
{
    return GET_DIM_W(dim_out);
}

int NN3dAvgPool::getOutputSize()
{
    return getOutputN() * getOutputC() * getOutputD() * getOutputH() * getOutputW();
}

void NN3dAvgPool::run(cudnnHandle_t cudnn_handle)
{
    float alpha = 1.f;
    float beta = 0.f;
    checkCUDNN(cudnnPoolingForward(cudnn_handle,
                                   desc_pool,
                                   &alpha,
                                   desc_in,
                                   data_input,
                                   &beta,
                                   desc_out,
                                   data_output));
}
