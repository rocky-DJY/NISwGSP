// cuda::匹配
#include "SurfCuda.h"
int GetMatchPointCount( const cv::Mat& data_img1, const cv::Mat& data_img2) {
	//指定使用的GPU序号，相关的还有下面几个函数可以使用
	 /* cv::cuda::getCudaEnabledDeviceCount();
	  cv::cuda::getDevice();
	  cv::cuda::DeviceInfo
	  cv::cuda::setDevice(0);*/

	/*向显存加载两张图片。这里需要注意两个问题：
	  第一，我们不能像操作（主）内存一样直接一个字节一个字节的操作显存，也不能直接从外存把图片加载到显存，一般需要通过内存作为媒介
	  第二，目前opencv的GPU SURF仅支持8位单通道图像，所以加上参数IMREAD_GRAYSCALE*/
	Mat img1,img2,imgdemo;
	//cv::cuda::GpuMat gmat1(3648, 5472, CV_8UC1);//创建一个加载图片的空gpumat
	//cv::cuda::GpuMat gmat2(3648, 5472, CV_8UC1);
	cv::cuda::GpuMat gmat1;
	cv::cuda::GpuMat gmat2;

	cv::cvtColor(data_img1,img1,cv::COLOR_BGR2GRAY);
	cv::cvtColor(data_img2,img2,cv::COLOR_BGR2GRAY);

	
	gmat1.upload(img1);
	gmat2.upload(img2);
	//ROI 块赋值
	//gmat01(cv::Rect(3087,513, 1193, 1130)).copyTo(gmat1(cv::Rect(3087, 513, 1193, 1130)));
	//gmat02(cv::Rect(2187,493, 1311, 1130)).copyTo(gmat2(cv::Rect(2187, 493, 1311, 1130)));
	//gmat01.copyTo(gmat1);
	//gmat02.copyTo(gmat2);
	/*gmat1.download(imgdemo);

	cv::namedWindow("img1", 0);
	cv::resizeWindow("img1", (int)(img1.size().width / 10), (int)(img1.size().height / 10));
	cv::imshow("img1", img1);

	cv::namedWindow("demo", 0);
	cv::resizeWindow("demo", (int)(img1.size().width / 10), (int)(img1.size().height / 10));
	cv::imshow("demo", imgdemo);
	cv::waitKey(0);
	cv::destroyAllWindows();*/

	/*下面这个函数的原型是：
	explicit SURF_CUDA(double
		_hessianThreshold, //SURF海森特征点阈值
		int _nOctaves=4, //尺度金字塔个数
		int _nOctaveLayers=2, //每一个尺度金字塔层数
		bool _extended=false, //如果true那么得到的描述子是128维，否则是64维
		float _keypointsRatio=0.01f,
		bool _upright = false
		);
	要理解这几个参数涉及SURF的原理*/
	cv::cuda::SURF_CUDA surf(80, 4, 3);
	/*分配下面几个GpuMat存储keypoint和相应的descriptor*/
	cv::cuda::GpuMat keypt1, keypt2;
	cv::cuda::GpuMat desc1, desc2;

	/*检测特征点*/
	surf(gmat1, cv::cuda::GpuMat(), keypt1, desc1);
	surf(gmat2, cv::cuda::GpuMat(), keypt2, desc2);

	/*匹配，下面的匹配部分和CPU的match没有太多区别,这里新建一个Brute-Force Matcher，一对descriptor的L2距离小于0.1则认为匹配*/
	auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
	//vector< vector< DMatch> > match_vec;
	vector<cv::DMatch> match_vec;
	matcher->match(desc1, desc2, match_vec);
	/*int count = 0;
	for (auto & d : match_vec) {
		if (d.distance < 0.1) 
			count++;
	}*/
	// downloading results  Gpu -> Cpu
	vector< KeyPoint> keypoints1, keypoints2;
	vector< float> descriptors1, descriptors2;
	surf.downloadKeypoints(keypt1, keypoints1);
	surf.downloadKeypoints(keypt2, keypoints2);
	surf.downloadDescriptors(desc1, descriptors1);
	surf.downloadDescriptors(desc2, descriptors2);

	int ptcount = (int)match_vec.size();
	Mat p1(ptcount, 2, CV_32F);
	Mat p2(ptcount, 2, CV_32F);

	//change keypoint to mat
	Point2f pt;
	for (int i = 0; i < ptcount; i++)
	{
		pt = keypoints1[match_vec[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keypoints2[match_vec[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	//use RANSAC to calculate F
	Mat fundamental;
	vector <uchar> RANSACStatus;
	fundamental = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC,10,0.99);
	//下面的代码为RANSAC优化后的特征点匹配效果
		//calculate the number of outliner
	int outlinerCount = 0;
	for (int i = 0; i < ptcount; i++)
	{
		if (RANSACStatus[i] == 0)
			outlinerCount++;
	}
	//calculate inLiner
	vector<Point2f> inliner1, inliner2;
	vector<DMatch> inlierMatches;
	int inlinerCount = ptcount - outlinerCount;
	inliner1.resize(inlinerCount);
	inliner2.resize(inlinerCount);
	inlierMatches.resize(inlinerCount);
	int inlinerMatchesCount = 0;
	for (int i = 0; i < ptcount; i++)
	{
		if ((RANSACStatus[i] != 0)) {
            inliner1[inlinerMatchesCount].x = p1.at<float>(i, 0);
            inliner1[inlinerMatchesCount].y = p1.at<float>(i, 1);
            inliner2[inlinerMatchesCount].x = p2.at<float>(i, 0);
            inliner2[inlinerMatchesCount].y = p2.at<float>(i, 1);
            inlierMatches[inlinerMatchesCount].queryIdx = inlinerMatchesCount;
            inlierMatches[inlinerMatchesCount].trainIdx = inlinerMatchesCount;
            inlinerMatchesCount++;
		}
	}
	inliner1.resize(inlinerMatchesCount);
	inliner2.resize(inlinerMatchesCount);
	return inliner1.size();
}
