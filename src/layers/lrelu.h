#pragma once

#include "runnable.h"

const int blockSize = 256;

/**
 * @brief Class for Leaky ReLU implementing the run method in place
 *
 */
class LReLU
{
public:
    /**
     * @brief in-place Leaky ReLU that wraps a CUDA kernel
     *
     * @param n number of elems in tensor
     * @param input
     */
    void run(int n, float *input);
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
class Sigmoid
{
public:
    /**
     * @brief in-place Sigmoid that wraps a CUDA kernel
     *
     * @param n number of elems in tensor
     * @param input
     */
    void run(int n, float *input);
};
