#pragma once
#include "layer_utils.h"
class Runnable
{
public:
    /**
     * @brief Runs the underlying NN Layers
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input,
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output !!Should run make this or should it be premade?
     * @param output start of the ouput tensor data
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
