#pragma once
#include "../includes.h"

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
 * @brief Gives tensor dimension data by parsing filename
 *
 * @param filename
 * @return TsrDims
 */
TsrDims filename2dims(std::string const &filename)
{
    TsrDims td;
    std::vector<int> dim = std::vector<int>();
    auto idx = filename.find("__");
    auto subStr = filename.substr(idx + 2);
    std::istringstream ss(subStr);
    char c;
    do
    {
        int val;
        ss >> val;
        dim.push_back(val);
    } while (ss >> c, c == 'x');
    td.dims = dim;
    td.layout = true;
    return td;
}

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
SizedArrayFloat readTensor2FloatBuffer(std::string const &fName)
{
    // Read tensor file into the filter array
    float *buf;
    int len;
    std::ifstream is;
    is.open(fName, std::ios::binary);
    is.seekg(0, std::ios::end);
    len = is.tellg();
    is.seekg(0, std::ios::beg);
    int numFloats = len / sizeof(float);
    buf = new float[numFloats];
    is.read((char *)buf, len);
    is.close();
    SizedArrayFloat sa;
    sa.arr = buf;
    sa.count = numFloats;
    return sa;
}
