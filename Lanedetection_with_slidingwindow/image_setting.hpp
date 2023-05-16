#include "opencv2/opencv.hpp"
#include "sliding_window.hpp"

extern double calibrate_mtx_data[9];
extern double dist_data[5];
extern cv::Mat calibrate_mtx;
extern cv::Mat distCoeffs;
extern cv::Mat cameraMatrix;
extern cv::Mat map1;
extern cv::Mat map2;
extern cv::Rect roi;


#pragma once
#ifndef IMAGE_SETTING_H
#define IMAGE_SETTING_H
class image_setting
{
public:
	image_setting() {
		cv::initUndistortRectifyMap(calibrate_mtx, distCoeffs, cv::Mat(), cameraMatrix, image_size, CV_32FC1, map1, map2);
	}

	~image_setting() {

	}
};
#endif