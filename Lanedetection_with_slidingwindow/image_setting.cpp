#include "opencv2/opencv.hpp"
#include "image_setting.hpp"

double calibrate_mtx_data[9] = {
350.354184, 0.0, 328.104147,
0.0, 350.652653, 236.540676,
0.0, 0.0, 1.0
};

double dist_data[5] = { -0.289296, 0.061035, 0.001786, 0.015238, 0.0 };

cv::Rect roi;
cv::Mat map1, map2;
cv::Mat calibrate_mtx(3, 3, CV_64FC1, calibrate_mtx_data);
cv::Mat distCoeffs(1, 4, CV_64FC1, dist_data);
cv::Mat cameraMatrix = getOptimalNewCameraMatrix(calibrate_mtx, distCoeffs, image_size, 1, image_size, &roi);