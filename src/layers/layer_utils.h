#pragma once
#include "../includes.h"

#define GET_DIM5_N(x) ((x).dims[0])
#define GET_DIM5_C(x) ((x).dims[1])
#define GET_DIM5_D(x) ((x).dims[2])
#define GET_DIM5_H(x) ((x).dims[3])
#define GET_DIM5_W(x) ((x).dims[4])

/**
 * @brief bundles tensor descriptor and start of tensor
 *
 */
typedef struct TensDescAndData
{
    cudnnTensorDescriptor_t desc;
    void *tens;
} TensDescAndData;
/**
 * @brief tag for tagged union
 *
 */
enum ExtraReturnTag
{
    NIL,
    TENSOR_DATA,
    LRELU_COEFF,
};
/**
 * @brief union for tagged union
 *
 */
typedef union ExtraRetVal
{
    void *nothing;
    TensDescAndData tensorBundle;
    float *slope;
} ExtraRetVal;

/**
 * @brief Nothing |
 * (TensorDescriptor, StartOfTensor) |
 * LReLU Coeff #[Feel free to add more to suit needs]
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
 * @brief Contains the dimensions of a 5D tensor or kernel, used with 3D convolutions
 * or pooling. Naturally, NCDHW for a tensor and C_out-C_in-DHW for a filter.
 */
typedef struct Dims5
{
    int dims[5];
} Dims5;

/**
 * @brief Contains the dimensions of a 4D tensor or kernel;
 * used with 2D convolutions or pooling
 */
typedef struct Dims4
{
    int dims[4];
} Dims4;

/**
 * @brief Contains an array of size 3, used for 3D convolution/pooling operations
 */
typedef struct Dims3
{
    int dims[3];
} Dims3;

/**
 * @brief Gives tensor dimension data by parsing filename
 *
 * @param filename
 * @return TsrDims
 */
TsrDims filename2dims(std::string const &filename);

/**
 * @brief Contains both the length of the array and a pointer to the first element
 *
 */
typedef struct SizedArrayFloat
{
    int count;
    float *arr;
} SizedArrayFloat;

/**
 * @brief Reads a tensor from file into array
 *
 * @param fName name of file
 * @return float* Pointer to float array. Caller is responsible for deleting
 */

/**
 * @brief Reads a tensor from file into an allocated array
 *
 * @param fName name of file
 * @return SizedArrayFloat Holds array size and pointer to data. Caller is responsible for deleting the array
 */
SizedArrayFloat readTensor2FloatBuffer(std::string const &fName);

// template <int numDims>
// typedef struct Dims
// {
//     int dims[numDims];
// } Dims;

Dims5 mkDims5(int d[5]);
Dims3 mkDims3(int d[3]);
