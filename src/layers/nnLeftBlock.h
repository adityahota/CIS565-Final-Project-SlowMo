#pragma once

#include "nn3dconv.h"
#include "nnGate.h"

// Combines orange and blue
class NNLeftBlock : public NNLayer
{
    // todo
private:
    NN3dConv orange;
    NNGate blue;
};
