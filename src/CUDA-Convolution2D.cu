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
	cv::Mat host_image_in = img_file_to_mat(file_name1);
	cv::Size s = host_image_in.size();
	int c = host_image_in.channels();
	std::cout << "size: " << s << " chan: " << c << std::endl;

	// Set the CUDA GPU device
	cudaSetDevice(0);

	// Create cuDNN handle
	cudnnHandle_t cudnn_handle;
	cudnnCreate(&cudnn_handle);

	// Set up tensor and filter dimensions
	int dim_N_in = 1, dim_C_in = 3, dim_D_in = 1, dim_H_in = 512, dim_W_in = 512;
	int kern_C_out = 3, kern_C_in = 3, kern_D = 1, kern_H = 1, kern_W = 1;

	// Create NN3dConv object for convolution
	NN3dConv *conv1 = new NN3dConv(dim_N_in, dim_C_in, dim_D_in, dim_H_in, dim_W_in,
	                               kern_C_out, kern_C_in, kern_D, kern_H, kern_W,
	                               0, 0, 0,
	                               1, 1, 1,
	                               1, 1, 1);
	std::cerr << "N: " << conv1->getOutputN() << " C: " << conv1->getOutputC() << " D: " << conv1->getOutputD()
              << " H: " << conv1->getOutputH() << " W: " << conv1->getOutputW() << std::endl;

    // Allocate space and copy data from input image to GPU
	// TODO: use loop so that tensor dimensions can be used later
    int bytes_image_in = dim_N_in * dim_C_in * dim_D_in * dim_H_in * dim_W_in * sizeof(float);
    float *dev_image_in = nullptr;
    cudaMalloc(&dev_image_in, bytes_image_in);
    cudaMemcpy(dev_image_in, host_image_in.ptr<float>(0), bytes_image_in, cudaMemcpyHostToDevice);

    // Allocate space for filter and initialize on host
    // TODO: use different library to allow for dynamic instantiation
    int i = 0;
    float host_filter[3][3][1][1][1];
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
                        host_filter[c_out][c_in][d][h][w] = 0.25f;
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
    float *rearranged = new float[bytes_image_out];

    for (int h = 0; h < 512; h++)
    {
        for (int w = 0; w < 512; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                // rearranged[c][h][w] = host_image_out[h][w][c];
                rearranged[c * 512 * 512 + h * 512 + w] = host_image_out[h * 512 * 3 + w * 3 + c];
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        std::cout << host_image_in.ptr<float>(0)[i] << " ";
        // std::cout << host_image_out[i] << " ";
    }
    std::cout << std::endl;


    mat_to_img_file(file_name2, host_image_out, conv1->getOutputH(), conv1->getOutputW());

    // Free allocated memory
      // delete[] host_image_out;
      cudaFree(dev_kernel);
      cudaFree(dev_image_in);
      cudaFree(dev_image_out);


//	// Allocate space and copy data from input image to GPU
//	float *dev_in_image = nullptr;
//	cudaMalloc(&dev_in_image, image_bytes);
//	cudaMemcpy(dev_in_image, in_image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);
//
//	// Allocate space for output image on GPU
//	float *dev_out_image = nullptr;
//	cudaMalloc(&dev_out_image, image_bytes);
//	cudaMemset(dev_out_image, 0, image_bytes);
//
//	// Specify the kernel template
//	const float kernel_template[3][3] = {
//		{1, 1, 1},
//		{1, -8, 1},
//		{1, 1, 1}};
//
//	// Copy kernel template into host buffer
//	float host_kernel[3][3][3][3];
//	for (int kernel = 0; kernel < 3; ++kernel)
//	{
//		for (int channel = 0; channel < 3; ++channel)
//		{
//			for (int row = 0; row < 3; ++row)
//			{
//				for (int column = 0; column < 3; ++column)
//				{
//					host_kernel[kernel][channel][row][column] = kernel_template[row][column];
//				}
//			}
//		}
//	}
//
//	// Allocate space and copy data from kernel to GPU
//	float *dev_kernel = nullptr;
//	cudaMalloc(&dev_kernel, sizeof(host_kernel));
//	cudaMemcpy(dev_kernel, host_kernel, sizeof(host_kernel), cudaMemcpyHostToDevice);
//
//	// Perform the convolution on the GPU
//	const float alpha = 1.0f, beta = 0.0f;
//	checkCUDNN(cudnnConvolutionForward(cudnn_handle,
//									   &alpha,
//									   input_descriptor,
//									   dev_in_image,
//									   kernel_descriptor,
//									   dev_kernel,
//									   convolution_descriptor,
//									   algorithm_perf.algo,
//									   dev_workspace,
//									   workspace_bytes,
//									   &beta,
//									   output_descriptor,
//									   dev_out_image));
//
//	// Allocate space and copy data from output image to host
//	float *out_image = new float[image_bytes];
//	cudaMemcpy(out_image, dev_out_image, image_bytes, cudaMemcpyDeviceToHost);
//
//	mat_to_img_file(file_name2, out_image, height, width);
//
//	delete[] out_image;
//	cudaFree(dev_kernel);
//	cudaFree(dev_in_image);
//	cudaFree(dev_out_image);
//	cudaFree(dev_workspace);

	cudnnDestroy(cudnn_handle);

	std::cerr << "Finished..." << std::endl;
	return 0;
}
