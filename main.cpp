//
//  main.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include <iostream>
#include "./Stitching/NISwGSP_Stitching.h"
#include "./Debugger/TimeCalculator.h"
#include <chrono>
using namespace std;
using namespace chrono;

int main(int argc, char **argv) {

    Eigen::initParallel(); /* remember to turn off "Hardware Multi-Threading */
    cout << "nThreads = " << Eigen::nbThreads() << endl;
    cout << "[#Images : " << argc - 1 << "]" << endl;

    //vector<string> path ={"1","2","3","4","5","6","7","8"};
    vector<string> path ={"6"};
    TimeCalculator timer;
    int core_threads =8;
    if (path.size()<8)
        core_threads = path.size();
// #pragma omp parallel for num_threads(core_threads)
    for(int i = 0; i < path.size(); ++i) {
        // out << "i = " << i << ", [Images : " << argv[i] << "]" << endl;
        timer.start();
        
        MultiImages multi_images(path[i], LINES_FILTER_WIDTH, LINES_FILTER_LENGTH);
        // MultiImages multi_images(datapath, LINES_FILTER_WIDTH, LINES_FILTER_LENGTH);

        
        NISwGSP_Stitching niswgsp(multi_images);
       
        /* 2D */
        //niswgsp.setWeightToAlignmentTerm(1);
        //niswgsp.setWeightToLocalSimilarityTerm(0.75);
        //niswgsp.setWeightToGlobalSimilarityTerm(6, 20, GLOBAL_ROTATION_2D_METHOD);
        //niswgsp.writeImage(niswgsp.solve(BLEND_AVERAGE), BLENDING_METHODS_NAME[BLEND_AVERAGE]);
        //niswgsp.writeImage(niswgsp.solve(BLEND_LINEAR),  BLENDING_METHODS_NAME[BLEND_LINEAR]);
        /* 3D */
        niswgsp.setWeightToAlignmentTerm(1);
        niswgsp.setWeightToLocalSimilarityTerm(0.75);
        niswgsp.setWeightToGlobalSimilarityTerm(6, 20, GLOBAL_ROTATION_3D_METHOD);
        // niswgsp.writeImage(niswgsp.solve(BLEND_AVERAGE), BLENDING_METHODS_NAME[BLEND_AVERAGE]);
        //cv::imshow("demo", niswgsp.solve(BLEND_LINEAR));
        //cv::waitKey(0);
        //cv::destroyWindow("demo");
        //auto start = system_clock::now();
        niswgsp.writeImage(niswgsp.solve(BLEND_LINEAR),  BLENDING_METHODS_NAME[BLEND_LINEAR]);

        //auto end = system_clock::now();
        //auto duration = duration_cast<microseconds>(end - start);
        //cout << "solve cost: "
        //    << 1000 * double(duration.count()) * microseconds::period::num / microseconds::period::den
        //    << "ms" << endl;

        timer.end("[NISwGSP] " + multi_images.parameter.file_name);
    }
    return 0;
}
