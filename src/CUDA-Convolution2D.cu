/*
 ============================================================================
 Name        : CUDA-Convolution2D.cu
 Author      : Aditya Hota
 Version     :
 Copyright   : 
 Description : cuDNN Testing with Convolutions
 ============================================================================
 */

#include "image_manipulation.h"
#include "includes.h"

#include <cudnn.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

int main(void)
{
	std::string file_name1 = "/home/laurelin/cis565/CIS565-Final-Project-SlowMo/img/hawk.png";
	std::string file_name2 = "/home/laurelin/cis565/CIS565-Final-Project-SlowMo/img/hawk_filtered.png";

	// Load the image
	cv::Mat in_image = img_file_to_mat(file_name1);

	// Set the CUDA GPU device
	cudaSetDevice(0);

	// Create cuDNN handle
	cudnnHandle_t cudnn_handle;
	cudnnCreate(&cudnn_handle);

	// Create input descriptor
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
										  /*format=*/CUDNN_TENSOR_NHWC,
										  /*dataType=*/CUDNN_DATA_FLOAT,
										  /*batch_size=*/1,
										  /*channels=*/3,
										  /*image_height=*/in_image.rows,
										  /*image_width=*/in_image.cols));

	// Create descriptor for convolution filter (kernel)
	cudnnFilterDescriptor_t kernel_descriptor;
	checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
										  /*dataType=*/CUDNN_DATA_FLOAT,
										  /*format=*/CUDNN_TENSOR_NCHW,
										  /*out_channels=*/3,
										  /*in_channels=*/3,
										  /*kernel_height=*/3,
										  /*kernel_width=*/3));

	// Create descriptor for convolution operation
	cudnnConvolutionDescriptor_t convolution_descriptor;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
											   /*pad_height=*/1,
											   /*pad_width=*/1,
											   /*vertical_stride=*/1,
											   /*horizontal_stride=*/1,
											   /*dilation_height=*/1,
											   /*dilation_width=*/1,
											   /*mode=*/CUDNN_CROSS_CORRELATION,
											   /*computeType=*/CUDNN_DATA_FLOAT));

	// Compute output dimensions and number of channels
	int batch_size = 0, channels = 0, height = 0, width = 0;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
													 input_descriptor,
													 kernel_descriptor,
													 &batch_size,
													 &channels,
													 &height,
													 &width));
	std::cerr << "Output Image: " << height << " x " << width << " x " << channels
			  << std::endl;

	// Create descriptor for the output (based on output image statistics)
	cudnnTensorDescriptor_t output_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
										  /*format=*/CUDNN_TENSOR_NHWC,
										  /*dataType=*/CUDNN_DATA_FLOAT,
										  /*batch_size=*/1,
										  /*channels=*/3,
										  /*image_height=*/in_image.rows,
										  /*image_width=*/in_image.cols));

	// Get the algorithm that cuDNN will use for the convolution
	cudnnConvolutionFwdAlgoPerf_t algorithm_perf;
	int returned_algorithms = 0;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle,
													  input_descriptor,
													  kernel_descriptor,
													  convolution_descriptor,
													  output_descriptor,
													  1,
													  &returned_algorithms,
													  &algorithm_perf));
	std::cerr << returned_algorithms << " algorithms returned" << std::endl;
	std::cerr << "Using algorithm " << algorithm_perf.algo << std::endl;

	// Get the workspace size required for the convolution (in Bytes)
	size_t workspace_bytes = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
													   input_descriptor,
													   kernel_descriptor,
													   convolution_descriptor,
													   output_descriptor,
													   algorithm_perf.algo,
													   &workspace_bytes));
	std::cerr << "Workspace size: " << workspace_bytes / 1048576.0 << "MB" << std::endl;

	// Allocate required space for workspace
	void *dev_workspace = nullptr;
	cudaMalloc(&dev_workspace, workspace_bytes);
	int image_bytes = batch_size * channels * height * width * sizeof(float);

	// Allocate space and copy data from input image to GPU
	float *dev_in_image = nullptr;
	cudaMalloc(&dev_in_image, image_bytes);
	cudaMemcpy(dev_in_image, in_image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

	// Allocate space for output image on GPU
	float *dev_out_image = nullptr;
	cudaMalloc(&dev_out_image, image_bytes);
	cudaMemset(dev_out_image, 0, image_bytes);

	// Specify the kernel template
	const float kernel_template[3][3] = {
		{1, 1, 1},
		{1, -8, 1},
		{1, 1, 1}};

	// Copy kernel template into host buffer
	float host_kernel[3][3][3][3];
	for (int kernel = 0; kernel < 3; ++kernel)
	{
		for (int channel = 0; channel < 3; ++channel)
		{
			for (int row = 0; row < 3; ++row)
			{
				for (int column = 0; column < 3; ++column)
				{
					host_kernel[kernel][channel][row][column] = kernel_template[row][column];
				}
			}
		}
	}

	// Allocate space and copy data from kernel to GPU
	float *dev_kernel = nullptr;
	cudaMalloc(&dev_kernel, sizeof(host_kernel));
	cudaMemcpy(dev_kernel, host_kernel, sizeof(host_kernel), cudaMemcpyHostToDevice);

	// Perform the convolution on the GPU
	const float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(cudnn_handle,
									   &alpha,
									   input_descriptor,
									   dev_in_image,
									   kernel_descriptor,
									   dev_kernel,
									   convolution_descriptor,
									   algorithm_perf.algo,
									   dev_workspace,
									   workspace_bytes,
									   &beta,
									   output_descriptor,
									   dev_out_image));

	// Allocate space and copy data from output image to host
	float *out_image = new float[image_bytes];
	cudaMemcpy(out_image, dev_out_image, image_bytes, cudaMemcpyDeviceToHost);

	mat_to_img_file(file_name2, out_image, height, width);

	delete[] out_image;
	cudaFree(dev_kernel);
	cudaFree(dev_in_image);
	cudaFree(dev_out_image);
	cudaFree(dev_workspace);

	cudnnDestroyTensorDescriptor(input_descriptor);
	cudnnDestroyTensorDescriptor(output_descriptor);
	cudnnDestroyFilterDescriptor(kernel_descriptor);
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);

	cudnnDestroy(cudnn_handle);

	return 0;
}
