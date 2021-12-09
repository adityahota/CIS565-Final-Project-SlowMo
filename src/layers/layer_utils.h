#pragma once
#include "../includes.h"

typedef struct TensDescAndData
{
    cudnnTensorDescriptor_t desc;
    void *tens;
} TensDescAndData;
enum ExtraReturnTag
{
    NIL,
    TENSOR_DATA,
};
typedef union ExtraRetVal
{
    void *nothing;
    TensDescAndData tensorBundle;
} ExtraRetVal;

/**
 * @brief Nothing | (TensorDescriptor, StartOfTensor) [Feel free to add more to suit needs]
 */
typedef struct TagUnionExtraRet
{
    ExtraReturnTag tag;
    ExtraRetVal val;
} TagUnionExtraRet;

/**
 * @brief Contains both layout and dimension data for a tensor;
 * layout not used for loading weights/biases from file
 *
 */
typedef struct TsrDims
{
    bool layout;           // True: NCHW/NCDHW; False: NHWC/whatever the 5d one is
    std::vector<int> dims; // size= 4, 5, for kernels, 1 for biases I think
} TsrDims;

/**
 * @brief Gives tensor dimension data by parsing filename
 *
 * @param filename
 * @return TsrDims
 */
TsrDims filename2dims(std::string filename);
