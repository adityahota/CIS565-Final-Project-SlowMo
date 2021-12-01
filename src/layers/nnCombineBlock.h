#pragma once

#include "nnRightBlock.h"
#include "nnLeftBlock.h"

// Combine (the funky plus symbol) so append/concat/add
class NNCombineBlock : public NNLayer
{
    // todo
private:
    NNRightBlock r;
    NNLeftBlock l;
    NNLayer mid;
};
