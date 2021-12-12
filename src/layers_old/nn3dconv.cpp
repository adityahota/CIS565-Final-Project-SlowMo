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

    // Get output tensor dimensions
    int dim_sizes[5];
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(desc_conv,
                                                     desc_in,
                                                     desc_kern,
                                                     5,
                                                     dim_sizes));
    dim_N_out = dim_sizes[0];
    dim_C_out = dim_sizes[1];
    dim_D_out = dim_sizes[2];
    dim_H_out = dim_sizes[3];
    dim_W_out = dim_sizes[4];

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
}

void NN3dConv::setData(float *input, void *weights, float **output)
{
    data_input = input;
    data_filter = weights;
    data_output = output;
}

int NN3dConv::getOutputN()
{
    // return dim_N_in;
    return dim_N_out;
}

int NN3dConv::getOutputC()
{
    // return kern_C_out;
    return dim_C_out;
}

int NN3dConv::getOutputD()
{
    // return dim_D_in;
    //    int top = dim_D_in + 2 * pad_D - dil_D * (kern_D - 1) - 1;
    //    return top / str_D + 1;
    return dim_D_out;
}

int NN3dConv::getOutputH()
{
    //    int top = dim_H_in + 2 * pad_H - dil_H * (kern_H - 1) - 1;
    //    return top / str_H + 1;
    return dim_H_out;
}

int NN3dConv::getOutputW()
{
    //    int top = dim_W_in + 2 * pad_W - dil_W * (kern_W - 1) - 1;
    //    return top / str_W + 1;
    return dim_W_out;
}

int NN3dConv::getOutputSize()
{
    return getOutputN() * getOutputC() * getOutputD() * getOutputH() * getOutputW();
}

void NN3dConv::run(cudnnHandle_t cudnn_handle)
{
    // Initialize the algorithm
    cudnnConvolutionFwdAlgoPerf_t algorithm_perf;
    int returned_algorithms = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle,
                                                      desc_in,
                                                      desc_kern,
                                                      desc_conv,
                                                      desc_out,
                                                      1,
                                                      &returned_algorithms,
                                                      &algorithm_perf));
    std::cerr << returned_algorithms << " algorithms returned. Using algo " << algorithm_perf.algo << std::endl;

    // Allocate workspace size required for the convolution (in Bytes)
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       desc_in,
                                                       desc_kern,
                                                       desc_conv,
                                                       desc_out,
                                                       algorithm_perf.algo,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << workspace_bytes / 1048576.0 << "MB" << std::endl;
    cudaMalloc(&cudnn_workspace, workspace_bytes);

    // Run the convolution
    const float alpha = 1.f, beta = 0.f;
    std::cerr << "Running the convolution..." << std::endl;
    checkCUDNN(cudnnConvolutionForward(cudnn_handle,
                                       &alpha,
                                       desc_in,
                                       data_input,
                                       desc_kern,
                                       data_filter,
                                       desc_conv,
                                       algorithm_perf.algo,
                                       cudnn_workspace,
                                       workspace_bytes,
                                       &beta,
                                       desc_out,
                                       data_output));
    std::cerr << "Finished running the convolution" << std::endl;

    cudaFree(cudnn_workspace);
}

// NN3dConv::~NN3dConv()
//{
//     return;
// }
