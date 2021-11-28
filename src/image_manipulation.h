/*
 * image_manipulation.h
 *
 *  Created on: Nov 23, 2021
 *      Author: aditya
 */

#pragma once

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>

#include <string>

/**
 * @brief Loads an image into a matrix from a specified file.
 * 		  The image is loaded into a 32-bit float matrix with 3 color channels.
 */
cv::Mat img_file_to_mat(const std::string &file_path);

/**
 * @brief Saves an image from a matrix into a specified file.
 */
void mat_to_img_file(const std::string &file_path, cv::Mat image);

/**
 * @brief Saves an image from a matrix into a specified file.
 */
void mat_to_img_file(const std::string &file_path,
		float *buffer, int height, int width);
