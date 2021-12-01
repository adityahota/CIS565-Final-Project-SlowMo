#pragma once

#include <iostream>
#include "../includes.h"

// #include "../nn_utils/matrix.hh"

// Intended as abstract, possibly templated for nn layer
class NNLayer
{
protected:
    // std::string name;

public:
    virtual ~NNLayer() = 0;
    virtual void run() = 0;

    // virtual Matrix &forward(Matrix &A) = 0;
    // virtual Matrix &backprop(Matrix &dZ, float learning_rate) = 0;

    // std::string getName() { return this->name; };
};

// inline NNLayer::~NNLayer() {}
