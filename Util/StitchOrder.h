//
// Created by nvidia on 8/20/21.
//

#ifndef NISWGSP_STITCHORDER_H
#define NISWGSP_STITCHORDER_H

#endif //NISWGSP_STITCHORDER_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/img_hash.hpp>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"	//SurfFeatureDetector实际在该头文件中
#include <opencv2/calib3d.hpp> /* CV_RANSAC */
#include <chrono>
#include "SurfCuda.h"
using namespace std;
using namespace cv;
using namespace chrono;
using namespace xfeatures2d;
using namespace cv::img_hash;
using namespace std;
vector<vector<bool>> Stitch_Order(vector<cv::Mat> &images);
int surf_match(const cv::Mat& img1, const cv::Mat& img2);
