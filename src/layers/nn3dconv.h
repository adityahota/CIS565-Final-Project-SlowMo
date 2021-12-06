#pragma once

#include "nnlayer.h"

// Orange in diagram
class NN3dConv : public NNLayer
{
private:
    int dim_N_in, dim_C_in, dim_D_in, dim_H_in, dim_W_in;
    int dim_N_out, dim_C_out, dim_D_out, dim_H_out, dim_W_out;
    int kern_C_out, kern_C_in, kern_D, kern_H, kern_W;
    int pad_D, pad_H, pad_W;
    int str_D, str_H, str_W;
    int dil_D, dil_H, dil_W;

    void *data_input;
    void *data_filter;
    void *data_output;
    void *cudnn_workspace;

    cudnnTensorDescriptor_t desc_in;
    cudnnTensorDescriptor_t desc_out;
    cudnnFilterDescriptor_t desc_kern;
    cudnnConvolutionDescriptor_t desc_conv;



public:
    NN3dConv(int N_in, int C_in, int D_in, int H_in, int W_in,
             int k_C_out, int k_C_in, int k_D, int k_H, int k_W,
             int p_D, int p_H, int p_W,
             int s_D, int s_H, int s_W,
             int d_D, int d_H, int d_W);
    ~NN3dConv();

    void run(cudnnHandle_t cudnn_handle);

    void setData(void *input, void *weights, void *output, void *cudnn_workspace);

    int getOutputN();
    int getOutputC();
    int getOutputD();
    int getOutputH();
    int getOutputW();
    int getOutputSize();

};
