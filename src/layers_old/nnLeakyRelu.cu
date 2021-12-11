

#include "nnLeakyRelu.h"

// __global__ void lReluKern(int n, float coeff, float *ptr)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= n)
//     {
//         return;
//     }
//     ptr[idx] *= ptr[idx] > 0.f ? 1.f : coeff;
// }
