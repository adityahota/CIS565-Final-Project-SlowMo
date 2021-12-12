#pragma once
#include "layer_utils.h"

/**
 * @brief Abstract class that implements a run function
 *
 */
class Runnable
{
public:
    /**
     * @brief Runs the underlying NN Layers
     *
     * @param h handle
     * @param inputDesc generally ignored; try to set descriptors at construction time
     * @param input input float tensor; passed in; self never owns; preallocated
     * @param outputDesc generally ignored; try to set descriptors at construction time
     * @param output tensor to store output; pointer passed in, self mallocs and passes forward
     * @param extra if any extra information needs to be passed along
     */
    virtual void run(cudnnHandle_t h,
                     cudnnTensorDescriptor_t const *inputDesc, void *input,
                     cudnnTensorDescriptor_t *outputDesc, void **output,
                     TagUnionExtraRet *extra) = 0;

    float one = 1.f;
    float zero = 0.f;
    float negOne = -1.f;
};
