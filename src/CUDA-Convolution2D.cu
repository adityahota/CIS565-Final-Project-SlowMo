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

#include <cudnn.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

int main(void)
{
	std::string file_name1 = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/img/apple.png";
	std::string file_name2 = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/img/apple_out.png";

	// Load the image
	cv::Mat host_image_in_NHWC = img_file_to_mat(file_name1);

	// Set the CUDA GPU device
	cudaSetDevice(0);

	// Create cuDNN handle
	cudnnHandle_t cudnn_handle;
	cudnnCreate(&cudnn_handle);

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
    int i = 0;
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
    cudaFree(dev_kernel);
    cudaFree(dev_image_in);
    cudaFree(dev_image_out);

	cudnnDestroy(cudnn_handle);

	std::cerr << "Finished..." << std::endl;
	return 0;
}
