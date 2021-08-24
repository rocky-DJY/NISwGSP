//
// Created by nvidai on 8/24/21.
//

#ifndef NISWGSP_SURFCUDA_H
#define NISWGSP_SURFCUDA_H

#endif //NISWGSP_SURFCUDA_H
#include <sstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <omp.h>
using namespace std;
using namespace cv;
using namespace cuda;
int GetMatchPointCount( const cv::Mat& data_img1, const cv::Mat& data_img2);
