#include "nn3dconv.h"

NN3dConv::NN3dConv(int N_in, int C_in, int D_in, int H_in, int W_in,
                   int k_C_out, int k_C_in, int k_D, int k_H, int k_W,
                   int p_D, int p_H, int p_W,
                   int s_D, int s_H, int s_W,
                   int d_D, int d_H, int d_W)
  : dim_N_in(N_in), dim_C_in(C_in), dim_D_in(D_in), dim_H_in(H_in), dim_W_in(W_in),
    kern_C_out(k_C_out), kern_C_in(k_C_in), kern_D(k_D), kern_H(k_H), kern_W(k_W),
    pad_D(p_D), pad_H(p_H), pad_W(p_W),
    str_D(s_D), str_H(s_H), str_W(s_W),
    dil_D(d_D), dil_H(d_H), dil_W(d_W)
{
    // Set data pointers to null
    data_input = nullptr;
    data_filter = nullptr;
    data_output = nullptr;
    cudnn_workspace = nullptr;


    // Create tensor, filter, and convolution descriptors
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_in));
    checkCUDNN(cudnnCreateTensorDescriptor(&desc_out));
    checkCUDNN(cudnnCreateFilterDescriptor(&desc_kern));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&desc_conv));

    // Set input tensor descriptor
    std::vector<int> v_dim_in = {dim_N_in, dim_C_in, dim_D_in, dim_H_in, dim_W_in};
    std::vector<int> v_str_in = {dim_C_in * dim_D_in * dim_H_in * dim_W_in,
                                 dim_D_in * dim_H_in * dim_W_in,
                                 dim_H_in * dim_W_in, dim_W_in, 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(desc_in,
                                          CUDNN_DATA_FLOAT,
                                          v_dim_in.size(),
                                          v_dim_in.data(),
                                          v_str_in.data()));

    // Set output tensor descriptor
    std::vector<int> v_dim_out = {getOutputN(), getOutputC(), getOutputD(), getOutputH(), getOutputW()};
    std::vector<int> v_str_out = {getOutputC() * getOutputD() * getOutputH() * getOutputW(),
                                  getOutputD() * getOutputH() * getOutputW(),
                                  getOutputH() * getOutputW(), getOutputW(), 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(desc_out,
                                          CUDNN_DATA_FLOAT,
                                          v_dim_out.size(),
                                          v_dim_out.data(),
                                          v_str_out.data()));

    // Set convolution descriptor
    std::vector<int> conv_pad_shape = {pad_D, pad_H, pad_W};
    std::vector<int> conv_str_shape = {str_D, str_H, str_W};
    std::vector<int> conv_dil_shape = {dil_D, dil_H, dil_W};
    checkCUDNN(cudnnSetConvolutionNdDescriptor(desc_conv,
                                               conv_pad_shape.size(),
                                               conv_pad_shape.data(),
                                               conv_str_shape.data(),
                                               conv_dil_shape.data(),
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));

    // Set filter descriptor
    std::vector<int> kern_shape = {kern_C_out, kern_C_in, kern_D, kern_H, kern_W};
    checkCUDNN(cudnnSetFilterNdDescriptor(desc_kern,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          kern_shape.size(),
                                          kern_shape.data()));
}

int NN3dConv::getOutputN()
{
    return dim_N_in;
}

int NN3dConv::getOutputC()
{
    return kern_C_out;
}

int NN3dConv::getOutputD()
{
    return dim_D_in;
}

int NN3dConv::getOutputH()
{
    int top = dim_H_in + 2 * pad_H - dil_H * (kern_H - 1) - 1;
    return top / str_H + 1;
}

int NN3dConv::getOutputW()
{
    int top = dim_W_in + 2 * pad_W - dil_W * (kern_W - 1) - 1;
    return top / str_W + 1;
}

int NN3dConv::getOutputSize()
{
    return getOutputN() * getOutputC() * getOutputD() * getOutputH() * getOutputW();
}

void NN3dConv::run()
{

}

NN3dConv::~NN3dConv()
{
    return;
}
