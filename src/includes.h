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
