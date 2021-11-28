/*
 * image_manipulation.cpp
 *
 *  Created on: Nov 23, 2021
 *      Author: aditya
 */

#include "image_manipulation.h"

cv::Mat img_file_to_mat(const std::string &file_path)
{
	cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void mat_to_img_file(const std::string &file_path, cv::Mat image)
{
	cv::threshold(image, image, 0.0, 0.0, cv::THRESH_TOZERO);
	cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
	image.convertTo(image, CV_8UC3);
	cv::imwrite(file_path, image);

	std::cerr << "Wrote output to " << file_path << std::endl;
}

void mat_to_img_file(const std::string &file_path,
		float *data, int height, int width)
{
	cv::Mat image = cv::Mat(height, width, CV_32FC3, data);
	mat_to_img_file(file_path, image);
}
