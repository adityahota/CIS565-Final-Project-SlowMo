#pragma once
#include "runnable.h"

class ReMean : Runnable
{
public:
    /**
     * @brief Adds back the rgb means to the colors
     *
     * @param h cudnnHandle passed to each
     * @param inputDesc tensor descriptor associated with input
     * @param input start of the input tensor data
     * @param outputDesc tensor descriptor associated with output
     * @param output start of the ouput tensor data
     * @param extra unused
     */
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void **output,
             TagUnionExtraRet *extra) override;
    /**
     * @brief Updates the rgb means since they are not known at construction time
     *
     * @param extra Tagged union of data passed out from a runnable;
     * should always be tensor bundle
     * @return true succeessfully set the fields
     * @return false unreachable; throws error beforehand
     */
    bool updateMeans(TagUnionExtraRet *extra);

private:
    cudnnTensorDescriptor_t avgTsrDesc;
    void *avgTsr;
};
