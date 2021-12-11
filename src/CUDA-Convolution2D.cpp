/*
 ============================================================================
 Name        : CUDA-Convolution2D.cu
 Author      : Aditya Hota
 Version     :
 Copyright   : 
 Description : cuDNN Testing with Convolutions
 ============================================================================
 */

#include "includes.h"
#include "image_manipulation.h"

#include "layers/nn3dconv.h"
#include "layers/nnLeakyRelu.h"
#include "layers/nn3dAvgPool.h"

#include <cudnn.h>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

void convolution3dTest(cudnnHandle_t cudnn_handle)
{
    std::string file_name1 = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/img/apple.png";
    std::string file_name2 = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/img/apple_out.png";

    // Load the image
    cv::Mat host_image_in_NHWC = img_file_to_mat(file_name1);

    // Set up tensor and filter dimensions
    int dim_N_in = 1, dim_C_in = 3, dim_D_in = 1, dim_H_in = 512, dim_W_in = 512;
    int kern_C_out = 3, kern_C_in = 3, kern_D = 1, kern_H = 2, kern_W = 2;

    // Create NN3dConv object for convolution
    NN3dConv *conv1 = new NN3dConv(dim_N_in, dim_C_in, dim_D_in, dim_H_in, dim_W_in,
                                   kern_C_out, kern_C_in, kern_D, kern_H, kern_W,
                                   0, 0, 0,
                                   1, 2, 3,
                                   1, 1, 1);
    std::cerr << "N: " << conv1->getOutputN() << " C: " << conv1->getOutputC() << " D: " << conv1->getOutputD()
              << " H: " << conv1->getOutputH() << " W: " << conv1->getOutputW() << std::endl;

    // Calculate space for input image and transform into NCHW format
    int bytes_image_in = dim_N_in * dim_C_in * dim_D_in * dim_H_in * dim_W_in * sizeof(float);
    float *host_image_in = new float[bytes_image_in];
    for (int h = 0; h < dim_H_in; h++)
    {
        for (int w = 0; w < dim_W_in; w++)
        {
            for (int c = 0; c < dim_C_in; c++)
            {
                host_image_in[c * dim_H_in * dim_W_in + h * dim_H_in + w] = host_image_in_NHWC.ptr<float>(0)[h * dim_W_in * dim_C_in + w * dim_C_in + c];
            }
        }
    }

    // Copy input image to GPU
    float *dev_image_in = nullptr;
    cudaMalloc(&dev_image_in, bytes_image_in);
    cudaMemcpy(dev_image_in, host_image_in, bytes_image_in, cudaMemcpyHostToDevice);

    // Allocate space for filter and initialize on host
    // TODO: use different library to allow for dynamic instantiation
    float host_filter[3][3][1][2][2];
    for (int c_out = 0; c_out < kern_C_out; c_out++)
    {
        for (int c_in = 0; c_in < kern_C_in; c_in++)
        {
            for (int d = 0; d < kern_D; d++)
            {
                for (int h = 0; h < kern_H; h++)
                {
                    for (int w = 0; w < kern_W; w++)
                    {
                        host_filter[c_out][c_in][d][h][w] = 0.05f;
                    }
                }
            }
        }
    }

    // Allocate space and copy data from host filter to GPU
    int bytes_kernel = kern_C_out * kern_C_in * kern_D * kern_H * kern_W * sizeof(float);
    float *dev_kernel = nullptr;
    cudaMalloc(&dev_kernel, bytes_kernel);
    cudaMemcpy(dev_kernel, host_filter, bytes_kernel, cudaMemcpyHostToDevice);

    // Allocate space for output image on GPU
    int bytes_image_out = conv1->getOutputN() * conv1->getOutputC() * conv1->getOutputD() * conv1->getOutputH() * conv1->getOutputW() * sizeof(float);
    float *dev_image_out = nullptr;
    cudaMalloc(&dev_image_out, bytes_image_out);
    cudaMemset(dev_image_out, 0, bytes_image_out);

    // Set data pointers for convolution
    conv1->setData(dev_image_in, dev_kernel, dev_image_out);

    // Run the convolution
    conv1->run(cudnn_handle);

    // Allocate space and copy data from image output on GPU to host
    float *host_image_out = new float[bytes_image_out];
    float *host_image_out_NHWC = new float[bytes_image_out];
    cudaMemcpy(host_image_out, dev_image_out, bytes_image_out, cudaMemcpyDeviceToHost);

    // Convert GPU output from NCHW to NHWC format
    for (int h = 0; h < conv1->getOutputH(); h++)
    {
        for (int w = 0; w < conv1->getOutputW(); w++)
        {
            for (int c = 0; c < conv1->getOutputC(); c++)
            {
                host_image_out_NHWC[h * conv1->getOutputW() * conv1->getOutputC() + w * conv1->getOutputC() + c] = host_image_out[c * conv1->getOutputH() * conv1->getOutputW() + h * conv1->getOutputW() + w];
            }
        }
    }

    for (int i = 0; i < 9; i++)
    {
        std::cout << host_image_out[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 9; i < 18; i++)
    {
        std::cout << host_image_out[i] << " ";
    }
    std::cout << std::endl;

    // Write the image to the file
    mat_to_img_file(file_name2, host_image_out_NHWC, conv1->getOutputH(), conv1->getOutputW());

    // Free allocated memory
    delete[] host_image_out;
    delete[] host_image_out_NHWC;
    delete conv1;
    cudaFree(dev_kernel);
    cudaFree(dev_image_in);
    cudaFree(dev_image_out);

    std::cerr << "Finished..." << std::endl;
}

void convolution3dTestDepth2(cudnnHandle_t cudnn_handle)
{
    std::string file_name1 = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/img/apple.png";

    // Load the image
    cv::Mat host_image_in_NHWC = img_file_to_mat(file_name1);

    // Set up tensor and filter dimensions
    int dim_N_in = 1, dim_C_in = 3, dim_D_in = 1, dim_H_in = 512, dim_W_in = 512;
    int kern_C_out = 3, kern_C_in = 3, kern_D = 2, kern_H = 1, kern_W = 5;

    // Create NN3dConv object for convolution
    NN3dConv *conv1 = new NN3dConv(dim_N_in, dim_C_in, dim_D_in, dim_H_in, dim_W_in,
                                   kern_C_out, kern_C_in, kern_D, kern_H, kern_W,
                                   1, 1, 1,
                                   1, 2, 3,
                                   1, 1, 1);
    std::cerr << "N: " << conv1->getOutputN() << " C: " << conv1->getOutputC() << " D: " << conv1->getOutputD()
              << " H: " << conv1->getOutputH() << " W: " << conv1->getOutputW() << std::endl;

    // Calculate space for input image and transform into NCHW format
    int bytes_image_in = dim_N_in * dim_C_in * dim_D_in * dim_H_in * dim_W_in * sizeof(float);
    float *host_image_in = new float[bytes_image_in / sizeof(float)];
    for (int h = 0; h < dim_H_in; h++)
    {
        for (int w = 0; w < dim_W_in; w++)
        {
            for (int c = 0; c < dim_C_in; c++)
            {
                host_image_in[c * dim_H_in * dim_W_in + h * dim_H_in + w] = host_image_in_NHWC.ptr<float>(0)[h * dim_W_in * dim_C_in + w * dim_C_in + c];
            }
        }
    }

    // Copy input image to GPU
    float *dev_image_in = nullptr;
    cudaMalloc(&dev_image_in, bytes_image_in);
    cudaMemcpy(dev_image_in, host_image_in, bytes_image_in, cudaMemcpyHostToDevice);

    // Allocate space for filter and initialize on host
    // TODO: use different library to allow for dynamic instantiation
    float host_filter[3][3][2][1][5];
    for (int c_out = 0; c_out < kern_C_out; c_out++)
    {
        for (int c_in = 0; c_in < kern_C_in; c_in++)
        {
            for (int d = 0; d < kern_D; d++)
            {
                for (int h = 0; h < kern_H; h++)
                {
                    for (int w = 0; w < kern_W; w++)
                    {
                        host_filter[c_out][c_in][d][h][w] = 0.15 * d * h + 0.2 * w;
                    }
                }
            }
        }
    }

    // Allocate space and copy data from host filter to GPU
    int bytes_kernel = kern_C_out * kern_C_in * kern_D * kern_H * kern_W * sizeof(float);
    float *dev_kernel = nullptr;
    cudaMalloc(&dev_kernel, bytes_kernel);
    cudaMemcpy(dev_kernel, host_filter, bytes_kernel, cudaMemcpyHostToDevice);

    // Allocate space for output image on GPU
    int bytes_image_out = conv1->getOutputN() * conv1->getOutputC() * conv1->getOutputD() * conv1->getOutputH() * conv1->getOutputW() * sizeof(float);
    float *dev_image_out = nullptr;
    cudaMalloc(&dev_image_out, bytes_image_out);
    cudaMemset(dev_image_out, 0, bytes_image_out);

    // Set data pointers for convolution
    conv1->setData(dev_image_in, dev_kernel, dev_image_out);

    // Run the convolution
    conv1->run(cudnn_handle);

    // Allocate space and copy data from image output on GPU to host
    float *host_image_out = new float[bytes_image_out];
    cudaMemcpy(host_image_out, dev_image_out, bytes_image_out, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; i++)
    {
        std::cout << host_image_out[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 9; i < 18; i++)
    {
        std::cout << host_image_out[i] << " ";
    }
    std::cout << std::endl;

    std::cout << host_image_out[83409] << std::endl;

    // Free allocated memory
    delete[] host_image_out;
    delete conv1;
    cudaFree(dev_kernel);
    cudaFree(dev_image_in);
    cudaFree(dev_image_out);

    std::cerr << "Finished..." << std::endl;
}

void leakyReLuTest(cudnnHandle_t cudnn_handle)
{
    int dim1 = 64, dim2 = 1, dim3 = 2, dim4 = 512, dim5 = 512;
    int num_elements = dim1 * dim2 * dim3 * dim4 * dim5;
    int num_elements_bytes = num_elements * sizeof(float);


    // Set up tensor on the host
    float *host_tensor = new float[num_elements];
    float *host_tensor_relu = new float[num_elements];
    for (int a = 0; a < dim1; a++)
    {
        for (int b = 0; b < dim2; b++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int d = 0; d < dim4; d++)
                {
                    for (int e = 0; e < dim5; e++)
                    {
                        int idx = a * dim2 * dim3 * dim4 * dim5 + b * dim3 * dim4 * dim5
                                  + c * dim4 * dim5 + d * dim5 + e;
                        host_tensor[idx] = pow(-1, e * a + c) * (0.15 * c * d + 0.2 * e);
                        // std::cout << idx << ": " << host_tensor[idx] << std::endl;
                    }
                }
            }
        }
    }

    // Copy tensor to the GPU
    float *dev_tensor = nullptr;
    cudaMalloc(&dev_tensor, num_elements_bytes);
    cudaMemcpy(dev_tensor, host_tensor, num_elements_bytes, cudaMemcpyHostToDevice);

    // Set up Leaky ReLU layer
    NNLeakyRelu *relu = new NNLeakyRelu(0.1);
    relu->setData(num_elements, dev_tensor, dev_tensor);

    // Run the Leaky ReLu kernel
    relu->run(cudnn_handle);

    // Copy tensor from the GPU
    cudaMemcpy(host_tensor_relu, dev_tensor, num_elements_bytes, cudaMemcpyDeviceToHost);

    // Print out all returned data
    for (int a = 0; a < dim1; a++)
    {
        for (int b = 0; b < dim2; b++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int d = 0; d < dim4; d++)
                {
                    for (int e = 0; e < dim5; e++)
                    {
                        int idx = a * dim2 * dim3 * dim4 * dim5 + b * dim3 * dim4 * dim5
                                  + c * dim4 * dim5 + d * dim5 + e;
                         // std::cout << idx << ": " << host_tensor_relu[idx] << std::endl;
                    }
                }
            }
        }
    }

    int index = 33554431;
    std::cout << index << ": " << host_tensor_relu[index] << std::endl;


    delete[] host_tensor;
    delete relu;
    cudaFree(dev_tensor);
    std::cerr << "Finished..." << std::endl;

}

void avgPool3dTest(cudnnHandle_t cudnn_handle)
{
    int dim1 = 3, dim2 = 3, dim3 = 2, dim4 = 1, dim5 = 5;
    int num_elements_in = dim1 * dim2 * dim3 * dim4 * dim5;
    int num_elements_in_bytes = num_elements_in * sizeof(float);

    // Set up tensor on the host
    float *host_tensor_in = new float[num_elements_in];
    for (int a = 0; a < dim1; a++)
    {
        for (int b = 0; b < dim2; b++)
        {
            for (int c = 0; c < dim3; c++)
            {
                for (int d = 0; d < dim4; d++)
                {
                    for (int e = 0; e < dim5; e++)
                    {
                        int idx = a * dim2 * dim3 * dim4 * dim5 + b * dim3 * dim4 * dim5
                                  + c * dim4 * dim5 + d * dim5 + e;
                        host_tensor_in[idx] = 0.15 * c * d + 0.2 * e + a * b;
                        // std::cout << idx << ": " << host_tensor[idx] << std::endl;
                    }
                }
            }
        }
    }


    // Set up 3D Avg Pool layer
    NN3dAvgPool *pool = new NN3dAvgPool(dim1, dim2, dim3, dim4, dim5,
                                        dim3, dim4, dim5,
                                        0, 0, 0,
                                        dim3, dim4, dim5);

    std::cerr << "N: " << pool->getOutputN() << " C: " << pool->getOutputC() << " D: " << pool->getOutputD()
              << " H: " << pool->getOutputH() << " W: " << pool->getOutputW() << std::endl;

    // Get output dimensions and allocate memory on host
    int num_elements_out = pool->getOutputN() * pool->getOutputC() *
                           pool->getOutputD() * pool->getOutputH() * pool->getOutputW();
    int num_elements_out_bytes = num_elements_out * sizeof(float);
    float *host_tensor_out = new float[num_elements_out];

    // Set up tensors on the device and copy data
    float *dev_tensor_in = nullptr;
    cudaMalloc(&dev_tensor_in, num_elements_in_bytes);
    cudaMemcpy(dev_tensor_in, host_tensor_in, num_elements_in_bytes, cudaMemcpyHostToDevice);

    float *dev_tensor_out = nullptr;
    cudaMalloc(&dev_tensor_out, num_elements_out_bytes);
    cudaMemset(dev_tensor_out, 0, num_elements_out_bytes);

    // Set Avg Pool operation data pointers
    pool->setData(dev_tensor_in, dev_tensor_out);

    // Run the Avg Pool operation
    pool->run(cudnn_handle);

    // Copy the data back to the host
    cudaMemcpy(host_tensor_out, dev_tensor_out, num_elements_out_bytes, cudaMemcpyDeviceToHost);

    // Print the data
    for (int i = 0; i < num_elements_out; i++)
    {
        std::cout << i << ": " << host_tensor_out[i] << std::endl;
    }

}

int main(void)
{
    std::cerr << "Starting..." << std::endl;
    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // Set the CUDA GPU device
    cudaSetDevice(0);

    // convolution3dTestDepth2(cudnn_handle);
    // leakyReLuTest(cudnn_handle);
    avgPool3dTest(cudnn_handle);

    std::cerr << "Exiting..." << std::endl;
    return 0;
}
