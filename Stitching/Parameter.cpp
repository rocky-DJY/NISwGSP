//
//  Parameter.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//
// #include <direct.h>
#include "Parameter.h"
/*#include <chrono>
using namespace chrono*/;
// #define NDEBUG     // �����ͼƬ��Ϣ
vector<string> getImageFileFullNamesInDir(const string & dir_name) {
    DIR *dir;
    struct dirent *ent;
    vector<string> result;
    
    const vector<string> image_formats = {
        ".bmp",  ".jpg" , ".JPG","png"
         };
    
    if((dir = opendir (dir_name.c_str())) != NULL) {
        while((ent = readdir (dir)) != NULL) {
            string file = string(ent->d_name);
            for(int i = 0; i < image_formats.size(); ++i) {
                if(file.length() > image_formats[i].length() &&
                   image_formats[i].compare(file.substr(file.length() - image_formats[i].length(),
                                                        image_formats[i].length())) == 0) {
                    result.emplace_back(file);
                }
            }
        }
        closedir(dir);
    } else {
        printError("F(getImageFileFullNamesInDir) could not open directory");
    }
    return result;
}

bool isFileExist (const string & name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

Parameter::Parameter(const string & _file_name) {

    
    file_name = _file_name;
    file_dir = "./input-42-data/" + _file_name + "/";
    result_dir = "./input-42-data/0_results/" + _file_name + "-result/";
    
//	_mkdir("./input-42-data/0_results/");
//	_mkdir(result_dir.c_str());
	mkdir("./input-42-data/0_results/", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    mkdir(result_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
#ifndef NDEBUG
    debug_dir = "./input-42-data/1_debugs/" + _file_name + "-result/";
//	_mkdir("./input-42-data/1_debugs/");
//	_mkdir(debug_dir.c_str());
	mkdir("./input-42-data/1_debugs/", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    mkdir(debug_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
#endif
    
    stitching_parse_file_name = file_dir + _file_name + "-STITCH-GRAPH.txt";

    image_file_full_names = getImageFileFullNamesInDir(file_dir);
//    for(int i=0;i<5;i++){
//        string s = to_string(i) + ".JPG";
//        image_file_full_names.push_back(s);
//    }
    /*** configure ***/
    grid_size = GRID_SIZE;
    down_sample_image_size = DOWN_SAMPLE_IMAGE_SIZE;
    
    /*** by file ***/
    if(isFileExist(stitching_parse_file_name)) {
        InputParser input_parser(stitching_parse_file_name);
        
        global_homography_max_inliers_dist   = input_parser.get<double>("*global_homography_max_inliers_dist"  , &GLOBAL_HOMOGRAPHY_MAX_INLIERS_DIST);
        local_homogrpahy_max_inliers_dist    = input_parser.get<double>( "*local_homogrpahy_max_inliers_dist"  , &LOCAL_HOMOGRAPHY_MAX_INLIERS_DIST);
        local_homography_min_features_count  = input_parser.get<   int>( "*local_homography_min_features_count", &LOCAL_HOMOGRAPHY_MIN_FEATURES_COUNT);
        images_count                 = input_parser.get<   int>("images_count");
        center_image_index           = input_parser.get<   int>("center_image_index");
        center_image_rotation_angle  = input_parser.get<double>("center_image_rotation_angle");
        
        /*** check ***/
        
        assert(image_file_full_names.size() == images_count);
        assert(center_image_index >= 0 && center_image_index < images_count);
        /*************/
        
        images_match_graph_manually.resize(images_count);
        for(int i = 0; i < images_count; ++i) {
            images_match_graph_manually[i].resize(images_count, false);
            vector<int> labels = input_parser.getVec<int>("matching_graph_image_edges-"+to_string(i), false);
            for(int j = 0; j < labels.size(); ++j) {
                images_match_graph_manually[i][labels[j]] = true;
            }
        }
        
        /*** check ***/
        queue<int> que;
        vector<bool> label(images_count, false);
        que.push(center_image_index);
        while(que.empty() == false) {
            int n = que.front();
            que.pop();
            label[n] = true;
            for(int i = 0; i < images_count; ++i) {
                if(!label[i] && (images_match_graph_manually[n][i] || images_match_graph_manually[i][n])) {
                    que.push(i);
                }
            }
        }
        assert(std::all_of(label.begin(), label.end(), [](bool i){return i;}));
        
        /*************/
     
#ifndef NDEBUG
        cout << "center_image_index = " << center_image_index << endl;
        cout << "center_image_rotation_angle = " << center_image_rotation_angle << endl;
        cout << "images_count = " << images_count << endl;
        cout << "images_match_graph_manually = " << endl;
        for(int i = 0; i < images_match_graph_manually.size(); ++i) {
            for(int j = 0; j < images_match_graph_manually[i].size(); ++j) {
                cout << images_match_graph_manually[i][j] << " ";
            }
            cout << endl;
        }
#endif
    }
    /*** by auto ***/
//    else{
//        center_image_index = 0;
//        images_count = image_file_full_names.size();
//        center_image_rotation_angle = 0;
//        for(auto ele:image_file_full_names)
//            cout<<ele<<endl;
//    }
}

const vector<vector<bool> > & Parameter::getImagesMatchGraph() const{
    //cout<<"check: "<<images_match_graph_manually.empty()<<endl;
    if(1) {
        //cout<<"mmt"<<endl;
        /*** check ***/
        queue<int> que;
        vector<bool> label(images_count, false);
        que.push(center_image_index);
        while(que.empty() == false) {
            int n = que.front();
            que.pop();
            label[n] = true;
            for(int i = 0; i < images_count; ++i) {
                if(!label[i] && (images_match_graph_automatically_[n][i] || images_match_graph_automatically_[i][n])) {
                    que.push(i);
                }
            }
        }
        assert(std::all_of(label.begin(), label.end(), [](bool i){return i;}));
        // printError("F(getImagesMatchGraph) image match graph verification [2] didn't be implemented yet");
        return images_match_graph_automatically_; /* TODO */
    }
    return images_match_graph_manually;
}

const vector<pair<int, int> >  Parameter::getImagesMatchGraphPairList() const {
    cout << "demo " << endl;
    // cout<<images_match_graph_pair_list.size()<<endl;
    if(images_match_graph_pair_list.empty()) {
        const vector<vector<bool> > & images_match_graph = getImagesMatchGraph();   // 2ms
        
        for(int i = 0; i < images_match_graph.size(); ++i) {
            for(int j = 0; j < images_match_graph[i].size(); ++j) {
                if(images_match_graph[i][j]) {
                    images_match_graph_pair_list.emplace_back(i, j);
                }
            }
        }
    }
    return images_match_graph_pair_list;
}