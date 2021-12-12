#include "flavr.h"

__global__ void NHWC_2_NCHW(int n, float *in, float *out, int batch, int h, int w, int c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    auto in_c = idx;
    auto in_n = in_c / (h * w * c);
    in_c -= in_n * h * w * c;
    auto in_h = in_c / (w * c);
    in_c -= in_h * w * c;
    auto in_w = in_c / c;
    in_c -= in_w * c;
    auto outdex = in_w + w * in_h + w * h * in_c + w * h * c * in_n;
}
__global__ void NCHW_2_NHWC(int n, float *in, float *out, int batch, int h, int w, int c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    // TODO
    //  auto in_c = idx;
    //  auto in_n = in_c / (h * w * c);
    //  in_c -= in_n * h * w * c;
    //  auto in_h = in_c / (w * c);
    //  in_c -= in_h * w * c;
    //  auto in_w = in_c / c;
    //  in_c -= in_w * c;
    //  auto outdex = in_w + w * in_h + w * h * in_c + w * h * c * in_n;
}
__global__ void NHWC_2_NCDHW(int n, float *in, float *out, int batch, int c);
