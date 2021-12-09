#pragma once
#include "runnable.h"
#include "conv3d.h"

class Gate : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        // Do not mutate input
        // Pool cudnnPoolingForward()
        // fcLayer.run()
        // Sigmoid cudnnActivationForward()
        // Multiply cudnnOpTensor() output = input * output of sigmoid
    }

private:
    Conv3d fcLayer;
};

class BStem : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override;
};
class BBlock : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override;
};
class DecBlock : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override;
};

class UNet : Runnable
{
public:
    void run(cudnnHandle_t h,
             cudnnTensorDescriptor_t const *inputDesc, void *input,
             cudnnTensorDescriptor_t *outputDesc, void *output,
             TagUnionExtraRet *extra) override
    {
        // enc0.run(input->x0);
        // enc1.run(x0->x1);
        // enc2.run(x1->x2);
        // enc3.run(x2->x3);
        // enc4.run(x3->x4);
        // dec0.run(x4->dx3);
        // dec1.run(xx3->dx2);
        // dec2.run(xx2->dx1);
        // dec3.run(xx1->dx0);
        // dec4.run(xx0->output);
    }

private:
    BStem enc0;
    BBlock enc1, enc2, enc3, enc4; //! Must not mutate input
    DecBlock dec0, dec1, dec2, dec3, dec4;

    //! NOTE: no easy cudnn to concat; format is NCDHW, cat along C,
    // Ex: xx0: alloc [1][128=64+64][D][H][W],
    // dx0 = &(xx0[0][0][0][0][0]), x0 = &(xx0[0][64][0][0][0])
    void *xx0, *xx1, *xx2, *xx3;
    void *x0, *x1, *x2, *x3, *x4;
    void *dx0, *dx1, *dx2, *dx3;
};
