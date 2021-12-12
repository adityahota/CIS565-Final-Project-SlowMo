#include "unmean.h"

void UnMean::run(cudnnHandle_t h,
                 cudnnTensorDescriptor_t const *inputDesc, float *input,
                 cudnnTensorDescriptor_t *outputDesc, float **output,
                 TagUnionExtraRet *extra)
{
    checkCUDNN(cudnnReduceTensor(h, tensReduceDescT,
                                 NULL, 0,
                                 wkspc, wkspcSize, // todo
                                 &one, inDescT, input,
                                 &zero, rgbAvgsDescT, rgbAvgs));
    checkCUDNN(cudnnAddTensor(h,
                              &negOne, rgbAvgsDescT, rgbAvgs,
                              &one, inDescT, input));
    // Assuming everything is pass by value'ed, since extras is a class field, it should live long enough
    {
        auto rgbBundle = TensDescAndData{rgbAvgsDescT, rgbAvgs};
        ExtraRetVal onion;
        onion.tensorBundle = rgbBundle;
        extras.tag = TENSOR_DATA;
        extras.val = onion; // maybe errors with stack, moving, ownership, lifetimes
        *extra = extras;
    }
    *outputDesc = inDescT;
    *output = input;
}
