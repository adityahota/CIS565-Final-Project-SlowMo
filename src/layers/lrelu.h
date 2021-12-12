#pragma once

#include "runnable.h"

const int blockSize = 256;

/**
 * @brief Class for Leaky ReLU implementing the run method in place
 *
 */
class LReLU : Runnable
{
public:
    /**
     * @brief Concrete run for in-place Leaky ReLU that wraps a CUDA kernel
     *
     * @param h -
     * @param inputDesc USED; PASS THIS IN
     * @param input
     * @param outputDesc unused
     * @param output Points back to the input as this operation is in-place
     * @param extra unused
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;
    /**
     * @brief Construct a new LReLU object
     *
     * @param coeff coefficient for when the value < 0
     */
    LReLU(float coeff) : coeff(coeff){};
    /**
     * @brief Default Constructor for normal ReLU
     *
     */
    LReLU() : LReLU(0.f){};

private:
    float coeff;
};

/**
 * @brief Class for sigmoid activation; runs in place
 *
 */
class Sigmoid : Runnable
{
public:
    /**
     * @brief Concrete run for in-place Sigmoid that wraps a CUDA kernel
     *
     * @param h -
     * @param inputDesc USED; PASS THIS IN
     * @param input
     * @param outputDesc unused
     * @param output Points back to the input as this operation is in-place
     * @param extra unused
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, float *input,
             cudnnTensorDescriptor_t *outputDesc, float **output,
             TagUnionExtraRet *extra) override;
};
