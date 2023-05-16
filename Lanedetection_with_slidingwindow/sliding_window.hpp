#include "opencv2/opencv.hpp"

extern cv::Size image_size;
extern std::vector<cv::Point2f> r_pos, l_pos;

#pragma once
#ifndef SLIDING_WINDOW_H
#define SLIDING_WINDOW_H
class SlidingWindow
{
public:
	cv::Mat calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi);
	cv::Mat warp_image(cv::Mat image);
	std::vector<cv::Point2f> warp_point(std::vector<cv::Point2f> point);
	cv::Mat binary_image_with_adaptivethreshold(cv::Mat image);
	cv::Mat binary_image_with_threshold(cv::Mat image);
	cv::Mat draw_sliding_window(cv::Mat image, int frame);
	cv::Mat contrast_clihe(cv::Mat image);
	cv::Mat morphological_transformation(cv::Mat image);
};
#endif