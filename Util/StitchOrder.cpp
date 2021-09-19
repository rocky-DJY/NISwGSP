#include "StitchOrder.h"
#define MATRIX
int surf_match(const cv::Mat& img1, const cv::Mat& img2)
{
    //step1: Detect the keypoints using SURF Detector
    int minHessian = 80;   // 数值越大 点数越少 耗时短
    //SurfFeatureDetector detector(minHessian);
    cv::Ptr<SURF> detector = SURF::create(minHessian);
    vector<cv::KeyPoint> keypoints1, keypoints2;
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);
    cv::Ptr<SURF> extractor = SURF::create(minHessian);
    cv::Mat descriptors1, descriptors2;
    extractor->compute(img1, keypoints1, descriptors1);
    extractor->compute(img2, keypoints2, descriptors2);
    //step3:Matching descriptor vectors with a brute force matcher
    cv::BFMatcher matcher(cv::NORM_L2, true);
    vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    int ptcount = (int)matches.size();
    cv::Mat p1(ptcount, 2, CV_32F);
    cv::Mat p2(ptcount, 2, CV_32F);
    //change keypoint to mat
    cv::Point2f pt;
    vector<cv::Point2f> p01, p02;
    for (int i = 0; i < ptcount; i++)
    {
        pt = keypoints1[matches[i].queryIdx].pt;
        p01.push_back(pt);
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;

        pt = keypoints2[matches[i].trainIdx].pt;
        p02.push_back(pt);
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }
    //use RANSAC to calculate F
    cv::Mat fundamental;
    vector <uchar> RANSACStatus;
    fundamental = findFundamentalMat(p1, p2, RANSACStatus, cv::FM_RANSAC);
    //下面的代码为RANSAC优化后的特征点匹配效果
    //calculate the number of outliner
    int outlinerCount = 0;
    for (int i = 0; i < ptcount; i++)
    {
        if (RANSACStatus[i] == 0)
            outlinerCount++;
    }
    //calculate inLiner
    vector<cv::Point2f> inliner1, inliner2;
    vector<cv::DMatch> inlierMatches;
    int inlinerCount = ptcount - outlinerCount;
    inliner1.resize(inlinerCount);
    inliner2.resize(inlinerCount);
    inlierMatches.resize(inlinerCount);
    int inlinerMatchesCount = 0;
    for (int i = 0; i < ptcount; i++)
    {
        if (RANSACStatus[i] != 0)
        {
            inliner1[inlinerMatchesCount].x = p1.at<float>(i, 0);
            inliner1[inlinerMatchesCount].y = p1.at<float>(i, 1);
            inliner2[inlinerMatchesCount].x = p2.at<float>(i, 0);
            inliner2[inlinerMatchesCount].y = p2.at<float>(i, 1);
            inlierMatches[inlinerMatchesCount].queryIdx = inlinerMatchesCount;
            inlierMatches[inlinerMatchesCount].trainIdx = inlinerMatchesCount;
            inlinerMatchesCount++;
        }
    }
    vector<cv::KeyPoint> key1(inlinerMatchesCount);
    vector<cv::KeyPoint> key2(inlinerMatchesCount);
    cv::KeyPoint::convert(inliner1, key1);
    cv::KeyPoint::convert(inliner2, key2);
    cv::Mat out;
    cv::Mat uvtrans = cv::Mat_<cv::Point2f>(inliner1.size(), 2);
    //vector<Point2f> p001, p002;
    // cv::Mat H = findHomography(inliner1, inliner2, cv::RANSAC);
    return inliner1.size();
}
template <typename T>
inline double test_one(const Mat &a, const Mat &b)
{
    Mat hashA, hashB;
    Ptr<ImgHashBase> func;
    func = T::create();
    func->compute(a, hashA);
    func->compute(b, hashB);
    return func->compare(hashA, hashB);
}

vector<vector<bool>> Stitch_Order(vector<cv::Mat> &images)
{
    auto start = system_clock::now();
    int s_rect = images[0].rows * images[0].cols;  //图像面积
    int dimension = 3200;

    vector<vector<bool>> res(images.size(),vector<bool>(images.size(),false));
    vector<vector<int>> dis(images.size(),vector<int>(images.size(),0));
    vector<pair<int,int>> image_parallel;
    vector<vector<vector<cv::Mat>>> image_space(images.size(),vector<vector<Mat>>(images.size()));
    for(int i=0;i<images.size();i++){
        for(int j=i+1; j<images.size();j++){
            vector<cv::Mat>  ele(images);
            image_space[i][j] = ele;
        }
    }
    for(int i = 0; i<images.size();i++){
        for(int j = 0;j<images.size();j++){
            Mat &input  = images[i];
            Mat &target = images[j];
            if(j>i){
                pair<int,int> temp = {i,j};
                image_parallel.push_back(temp);
            }
        }
    }
    //vector<pair<int,Mat>> addre(image_parallel.size());
 #pragma omp parallel for num_threads(8)
    for(int i =0;i<image_parallel.size();i++){
        pair<int,int> temp = image_parallel[i];
        Mat input = image_space[temp.first][temp.second][temp.first];
        Mat target = image_space[temp.first][temp.second][temp.second];

        // dis[temp.first][temp.second] = surf_match(input,target);
        // cuda surf //
        dis[temp.first][temp.second] = GetMatchPointCount(input,target);
    }
    // 设置阈值 //
    int threshold = s_rect / dimension;
    cout<<"s_threshold: "<<s_rect<<"  "<<threshold<<endl;
    for(int i =0;i<dis.size();i++){
        for(int j =i+1;j<dis.size();j++){
#ifdef MATRIX
            cout<<dis[i][j]<<"  ";
#endif
            if(dis[i][j]>threshold)
                res[i][j] = 1;
        }
#ifdef MATRIX
        cout<<endl;
#endif
    }
//    for(int i =0;i<res.size();i++){
//        for(int j=res.size()-1;j>i+1;j--){
//            if(res[i][j]){
//                // col serach
//                int flag0 = 0;
//                for(int k = i+1;k<j;k++){
//                    if(res[k][j]){
//                        flag0 = 1;
//                        break;
//                    }
//                }
//                // row search
//                int flag1 = 0;
//                for(int u = j-1;u>i;u--){
//                    if(res[i][u]){
//                        flag1=1;
//                        break;
//                    }
//                }
//                if(flag0 & flag1)
//                    res[i][j] = false;
//            }
//        }
//    }
    cout<<  "**---------------------*** "<<endl;
#ifdef MATRIX
    for(auto &ele:res){
        for(auto e:ele){
            cout<<e<<",";
        }
        cout<<endl;
    }
#endif
    auto end   = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
//    cout <<  "cost: "
//    << double(duration.count()) * microseconds::period::num / microseconds::period::den
//    << "s" << endl;
    return res;
}
