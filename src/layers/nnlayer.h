#pragma once

#include <iostream>
#include "../includes.h"

// Intended as abstract, possibly templated for nn layer
class NNLayer
{
protected:
    // std::string name;

public:
    virtual void run(cudnnHandle_t cudnn_handle) = 0;
};
