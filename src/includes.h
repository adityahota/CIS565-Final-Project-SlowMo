#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <string>
#include <vector>
#include <list>

#define BLOCK_SIZE 256

#define GET_DIM_N(x) (x[0])
#define GET_DIM_C(x) (x[1])
#define GET_DIM_D(x) (x[2])
#define GET_DIM_H(x) (x[3])
#define GET_DIM_W(x) (x[4])

typedef struct VidFrame
{ // TODO
} VidFrame;

// Error checking defines
#define checkCUDNN(expression)                                     \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }
