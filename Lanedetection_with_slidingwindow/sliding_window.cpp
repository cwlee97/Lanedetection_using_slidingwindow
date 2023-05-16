#include <numeric>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

#include "sliding_window.hpp"

constexpr int IMAGE_WIDTH = 640;
constexpr int IMAGE_HEIGHT = 480;
constexpr int HALF_WIDTH = 320;
constexpr int HALF_HEIGHT = 240;

cv::Size image_size = cv::Size(640, 480);
std::vector<cv::Point2f> r_pos, l_pos;

// calibrate 함수
cv::Mat SlidingWindow::calibrate_image(cv::Mat const& src, cv::Mat const& map1, cv::Mat const& map2, cv::Rect const& roi)
{
    // image calibrating
    cv::Mat mapping_image = src.clone();
    cv::Mat calibrated_image;
    remap(src, mapping_image, map1, map2, cv::INTER_LINEAR);

    // image slicing & resizing
    mapping_image = mapping_image(roi);
    resize(mapping_image, calibrated_image, image_size);

    return calibrated_image;
}

// warp 함수
cv::Mat SlidingWindow::warp_image(cv::Mat image)
{
    // Warping 관련 변수 선언
    int warp_image_width = HALF_WIDTH;
    int warp_image_height = HALF_HEIGHT;

    int warp_x_margin = 30;
    int warp_y_margin = 3;

    cv::Point src_pts1 = cv::Point2f(240 - warp_x_margin, HALF_HEIGHT + 40);
    cv::Point src_pts2 = cv::Point2f(0, IMAGE_HEIGHT - 20);
    cv::Point src_pts3 = cv::Point2f(400 + warp_x_margin, HALF_HEIGHT + 40);
    cv::Point src_pts4 = cv::Point2f(IMAGE_WIDTH, IMAGE_HEIGHT - 20);

    cv::Point dist_pts2 = cv::Point2f(0, warp_image_height);
    cv::Point dist_pts3 = cv::Point2f(warp_image_width, 0);
    cv::Point dist_pts4 = cv::Point2f(warp_image_width, warp_image_height);
    cv::Point dist_pts1 = cv::Point2f(0, 0);

    std::vector<cv::Point2f> warp_src_mtx = { src_pts1, src_pts2, src_pts3, src_pts4 };
    std::vector<cv::Point2f> warp_dist_mtx = { dist_pts1, dist_pts2, dist_pts3, dist_pts4 };

    cv::Mat src_to_dist_mtx = getPerspectiveTransform(warp_src_mtx, warp_dist_mtx);

    cv::Mat warped_image;
    warpPerspective(image, warped_image, src_to_dist_mtx, cv::Size(warp_image_width, warp_image_height), cv::INTER_LINEAR);

    // warp 기준점 확인
    circle(image, src_pts1, 20, cv::Scalar(255, 0, 0), -1);
    circle(image, src_pts2, 20, cv::Scalar(255, 0, 0), -1);
    circle(image, src_pts3, 20, cv::Scalar(255, 0, 0), -1);
    circle(image, src_pts4, 20, cv::Scalar(255, 0, 0), -1);

    return warped_image;
};

std::vector<cv::Point2f> SlidingWindow::warp_point(std::vector<cv::Point2f> point)
{
    int warp_image_width = HALF_WIDTH;
    int warp_image_height = HALF_HEIGHT;

    int warp_x_margin = 30;
    int warp_y_margin = 3;

    cv::Point src_pts1 = cv::Point2f(240 - warp_x_margin, HALF_HEIGHT + 40);
    cv::Point src_pts2 = cv::Point2f(0, IMAGE_HEIGHT - 20);
    cv::Point src_pts3 = cv::Point2f(400 + warp_x_margin, HALF_HEIGHT + 40);
    cv::Point src_pts4 = cv::Point2f(IMAGE_WIDTH, IMAGE_HEIGHT - 20);

    cv::Point dist_pts2 = cv::Point2f(0, warp_image_height);
    cv::Point dist_pts3 = cv::Point2f(warp_image_width, 0);
    cv::Point dist_pts4 = cv::Point2f(warp_image_width, warp_image_height);
    cv::Point dist_pts1 = cv::Point2f(0, 0);

    std::vector<cv::Point2f> warp_src_mtx = { src_pts1, src_pts2, src_pts3, src_pts4 };
    std::vector<cv::Point2f> warp_dist_mtx = { dist_pts1, dist_pts2, dist_pts3, dist_pts4 };
    cv::Mat dist_to_src_mtx = getPerspectiveTransform(warp_dist_mtx, warp_src_mtx);

    std::vector<cv::Point2f> warped_point;
    cv::perspectiveTransform(point, warped_point, dist_to_src_mtx);

    return warped_point;
}

// warp process image 함수
cv::Mat SlidingWindow::binary_image_with_adaptivethreshold(cv::Mat image)
{
    int lane_bin_th = 170;

    cv::Mat blur;
    GaussianBlur(image, blur, cv::Size(5, 5), 0);

    cv::Mat hls;
    cvtColor(blur, hls, cv::COLOR_BGR2HLS);
    std::vector<cv::Mat> L;
    split(hls, L);

    cv::Mat lane;
    cv::adaptiveThreshold(L[1], lane, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 201, 20);

    return lane;
}

cv::Mat SlidingWindow::binary_image_with_threshold(cv::Mat image)
{
    int lane_bin_th = 170;

    cv::Mat blur;
    GaussianBlur(image, blur, cv::Size(5, 5), 0);

    cv::Mat hls;
    cvtColor(blur, hls, cv::COLOR_BGR2HLS);
    std::vector<cv::Mat> L;
    split(hls, L);

    cv::Mat lane;
    threshold(L[1], lane, lane_bin_th, 255, cv::THRESH_BINARY_INV);

    return lane;
}

cv::Mat SlidingWindow::contrast_clihe(cv::Mat image)
{
    cv::Ptr<cv::CLAHE>clahe = cv::createCLAHE(0.2, cv::Size(8, 8));
    cv::Mat ycrcb_Mat;
    cv::cvtColor(image, ycrcb_Mat, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> ycrcb_planes(3);
    split(ycrcb_Mat, ycrcb_planes);

    cv::Mat clahe_y;
    clahe->apply(ycrcb_planes[0], clahe_y);
    clahe_y.copyTo(ycrcb_planes[0]);

    // convert back to BGR
    cv::Mat clahe_image;
    cv::merge(ycrcb_planes, ycrcb_Mat);
    cv::cvtColor(ycrcb_Mat, clahe_image, cv::COLOR_YCrCb2BGR);

    return clahe_image;
}

cv::Mat SlidingWindow::morphological_transformation(cv::Mat image)
{
    cv::Mat morphological_transformation_image;
    cv::Mat kernel = cv::Mat::ones(6, 6, CV_32F);
    cv::morphologyEx(image, morphological_transformation_image, cv::MORPH_OPEN, kernel);

    return morphological_transformation_image;
}

cv::Mat SlidingWindow::draw_sliding_window(cv::Mat image, int frame)
{
    int num_SlidingWindow = 18;
    int width_sliding_window = 20;

    int window_height = HALF_HEIGHT / num_SlidingWindow;
    int window_margin = 80;

    int min_points = window_height / 4;
    int min_pixels = 10;

    int lane_width = 200;

    std::vector<int> lx, ly, rx, ry;

    bool before_l_detected = true;
    bool before_r_detected = true;

    cv::Mat out_img;
    cv::cvtColor(image, out_img, cv::COLOR_GRAY2BGR);


    // 자른 window 갯수만큼 for loop
    for (int i = num_SlidingWindow - 1; i >= 0; i--)
    {
        int leftx_current, rightx_current;

        // window 생성
        cv::Rect roi(0, i * image.rows / num_SlidingWindow, image.cols, image.rows / num_SlidingWindow);
        cv::Mat window = image(roi);

        // histogram 생성
        cv::Mat histogram;
        cv::reduce(window, histogram, 0, cv::REDUCE_SUM, CV_32S);

        // 가로 / 2를 저장하는 변수 선언
        int mid_point = image.rows / 2;

        // 좌, 우 차선으로 인식 된 부분의 히스토그램 변수 선언
        cv::Mat left_histogram;
        cv::Mat right_histogram;

        int right_histogram_start, left_histogram_start;

        // 첫 window는 화면 중앙을 기준으로 좌, 우 차선을 나눈다.
        if (i == (num_SlidingWindow - 1))
        {
            left_histogram = histogram(cv::Range::all(), cv::Range(0, mid_point));
            right_histogram = histogram(cv::Range::all(), cv::Range(mid_point, histogram.cols - 1));

            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            if (left_max_val > 255 * min_points) leftx_current = left_max_loc.x;
            else leftx_current = NULL;

            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            if (right_max_val > 255 * min_points) rightx_current = right_max_loc.x;
            else rightx_current = NULL;

            right_histogram_start = mid_point;
            left_histogram_start = 0;
        }

        // 이전 window에서 차선을 둘 다 인식한 경우
        else if (before_l_detected == true && before_r_detected == true)
        {
            left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), std::min(histogram.cols - 1, lx.back() + window_margin)));
            right_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, rx.back() - window_margin), std::min(histogram.cols - 1, rx.back() + window_margin)));

            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            if (left_max_val > 255 * min_points) leftx_current = left_max_loc.x;
            else leftx_current = NULL;

            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            if (right_max_val > 255 * min_points) rightx_current = right_max_loc.x;
            else rightx_current = NULL;

            right_histogram_start = std::max(0, rx.back() - window_margin);
            left_histogram_start = std::max(0, lx.back() - window_margin);
        }

        // 이전 window에서 왼쪽 차선 인식 X, 오른쪽 차선만 인식 O한 경우
        else if (before_l_detected == false && before_r_detected == true)
        {
            if (rx.back() - lane_width > 0)
            {
                left_histogram = histogram(cv::Range::all(), cv::Range(0, rx.back() - lane_width));

                double left_max_val;
                cv::Point left_max_loc;
                cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
                if (left_max_val > 255 * min_points) leftx_current = left_max_loc.x;
                else leftx_current = NULL;
            }
            else
            {
                leftx_current = NULL;
            }

            right_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, rx.back() - window_margin), std::min(HALF_WIDTH - 1, rx.back() + window_margin)));

            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            if (right_max_val > 255 * min_points) rightx_current = right_max_loc.x;
            else rightx_current = NULL;

            right_histogram_start = std::max(0, rx.back() - window_margin);
            left_histogram_start = 0;
        }

        // 이전 window에서 왼쪽 차선은 인식 O, 오른쪽 차선은 인식 X한 경우
        else if (before_l_detected == true && before_r_detected == false)
        {
            if (lx.back() + lane_width < HALF_WIDTH)
            {
                right_histogram = histogram(cv::Range::all(), cv::Range(lx.back() + lane_width, histogram.cols - 1));

                double right_max_val;
                cv::Point right_max_loc;
                cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
                if (right_max_val > 255 * min_points) rightx_current = right_max_loc.x;
                else rightx_current = NULL;
            }
            else
            {
                rightx_current = NULL;
            }
            left_histogram = histogram(cv::Range::all(), cv::Range(std::max(0, lx.back() - window_margin), std::min(histogram.cols - 1, lx.back() + window_margin)));

            double left_max_val;
            cv::Point left_max_loc;

            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            if (left_max_val > 255 * min_points) leftx_current = left_max_loc.x;
            else leftx_current = NULL;

            right_histogram_start = lx.back() + lane_width;
            left_histogram_start = std::max(0, lx.back() - window_margin);
        }

        // 이전 window에서 차선을 둘 다 인식하지 못한 경우
        else if (before_l_detected == false && before_r_detected == false)
        {
            left_histogram = histogram(cv::Range::all(), cv::Range(0, mid_point));
            right_histogram = histogram(cv::Range::all(), cv::Range(mid_point, histogram.cols - 1));

            double left_max_val;
            cv::Point left_max_loc;
            cv::minMaxLoc(left_histogram, NULL, &left_max_val, NULL, &left_max_loc);
            if (left_max_val > 255 * min_points) leftx_current = left_max_loc.x;
            else leftx_current = NULL;

            double right_max_val;
            cv::Point right_max_loc;
            cv::minMaxLoc(right_histogram, NULL, &right_max_val, NULL, &right_max_loc);
            if (right_max_val > 255 * min_points) rightx_current = right_max_loc.x;
            else rightx_current = NULL;

            right_histogram_start = mid_point;
            left_histogram_start = 0;
        }


        int win_yl = (i + 1) * window_height;
        int win_yh = i * window_height;

        // 오른쪽 차선
        if (rightx_current != NULL)
        {
            // 오른쪽에서 흰색 찾기
            cv::Mat right_nz;
            cv::findNonZero(right_histogram, right_nz);
            std::vector<int> right_nonzeros;

            for (int i = 0; i < right_nz.total(); i++)
            {
                right_nonzeros.push_back(right_nz.at<cv::Point>(i).x + right_histogram_start);
            }

            // 차선 픽셀 정보 저장
            // nz에서 지정 범위내에 있는 좌표값들만 차선이라고 인식
            // good_right_inds에 저장

            // 검출된 차선의 픽셀이 최소 픽셀의 개수보다 많은지
            if (right_nz.rows > min_pixels)
            {
                // 흰색 차선의 평균값을 right_current로 지정
                rightx_current = int(std::accumulate(right_nonzeros.begin(), right_nonzeros.end(), 0.0) / right_nonzeros.size());

                int win_xrl = rightx_current - width_sliding_window;
                int win_xrh = rightx_current + width_sliding_window;

                // 오른쪽 슬라이딩 윈도우 그리기
                cv::rectangle(out_img, cv::Point(win_xrl, win_yl), cv::Point(win_xrh, win_yh), cv::Scalar(0, 255, 0), 2);
                before_r_detected = true;
                if (frame % 30 == 0 && i == num_SlidingWindow - 6) r_pos.push_back(cv::Point((win_xrl + win_xrh) / 2, 400));
            }
            // 최소 픽셀 개수보다 검출한 차선의 픽셀 개수가 적다면
            else before_r_detected = false;

            // right_current가 담긴 lx 생성
            rx.push_back(rightx_current);
        }
        else before_r_detected = false;

        // 왼쪽 차선
        if (leftx_current != NULL)
        {
            // 왼쪽에서 흰색 찾기
            cv::Mat left_nz;

            cv::findNonZero(left_histogram, left_nz);
            std::vector<int> left_nonzeros;

            for (int i = 0; i < left_nz.total(); i++)
            {
                left_nonzeros.push_back(left_nz.at<cv::Point>(i).x + left_histogram_start);
            }

            // 검출된 차선의 픽셀이 최소 픽셀의 개수보다 많은지
            if (left_nz.rows > min_pixels)
            {
                // 흰색 차선의 평균값을 right_current로 지정
                leftx_current = int(std::accumulate(left_nonzeros.begin(), left_nonzeros.end(), 0.0) / left_nonzeros.size());

                // 슬라이딩을 그리기 위한 오른쪽 x좌표의 low, high값
                int win_xll = leftx_current - width_sliding_window;
                int win_xlh = leftx_current + width_sliding_window;

                // 오른쪽 슬라이딩 윈도우 그리기
                cv::rectangle(out_img, cv::Point(win_xll, win_yl), cv::Point(win_xlh, win_yh), cv::Scalar(0, 0, 255), 2);
                before_l_detected = true;
                if (frame % 30 == 0 && i == num_SlidingWindow - 6) l_pos.push_back(cv::Point((win_xll + win_xlh) / 2, 400));
            }
            // 최소 픽셀 개수보다 검출한 차선의 픽셀 개수가 적다면
            else before_l_detected = false;

            // right_current가 담긴 lx 생성
            lx.push_back(leftx_current);

        }
        else before_l_detected = false;
    }
    return out_img;
}