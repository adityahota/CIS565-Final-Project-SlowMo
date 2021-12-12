#include "remean.h"

void ReMean::run(cudnnHandle_t h,
                 cudnnTensorDescriptor_t const *inputDesc, float *input,
                 cudnnTensorDescriptor_t *outputDesc, float **output,
                 TagUnionExtraRet *extra)
{
    checkCUDNN(cudnnAddTensor(h,
                              &negOne, avgTsrDesc, avgTsr,
                              &one, *inputDesc, input));
}

bool ReMean::updateMeans(TagUnionExtraRet *extra)
{
    if (extra->tag == TENSOR_DATA)
    {
        avgTsrDesc = extra->val.tensorBundle.desc;
        avgTsr = extra->val.tensorBundle.tens;
        return true;
    }
    throw std::invalid_argument("Data Passed In not tensor bundle");
    return false;
}
