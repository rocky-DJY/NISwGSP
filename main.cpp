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

    string datapath = "APAP-train";
    TimeCalculator timer;
    for(int i = 1; i < 2; ++i) {
        // out << "i = " << i << ", [Images : " << argv[i] << "]" << endl;
        timer.start();
        
        MultiImages multi_images(argv[1], LINES_FILTER_WIDTH, LINES_FILTER_LENGTH);
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
