cmake_minimum_required(VERSION 3.10)

PROJECT(NISwGSP)

SET(CMAKE_BUILD_TYPE "Release")
FIND_PACKAGE(OpenMP REQUIRED)

#eigen
include_directories("/usr/include/eigen3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE -std=c++11")

#opencv
set(CMAKE_PREFIX_PATH   "/home/nvidai/DJY/opencv347_with_cuda/share/OpenCV")
find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})

#vlfeat
# FIND_LIBRARY(VLFEAT_LIBRARY libvl)
# link_directories("/home/nvidia/DJY/project/NISwGSP/vlfeat-0.9.20/bin/glnxa64")

aux_source_directory(./Debugger DEBUGGER_SRC)
aux_source_directory(./Feature FEATURE_SRC)
aux_source_directory(./Mesh MESH_SRC)
aux_source_directory(./Stitching STITCHING_SRC)
aux_source_directory(./Util UTIL_SRC)

add_executable(NISwGSP Configure.h main.cpp lsd.cpp lsd.h precomp.hpp ${DEBUGGER_SRC} ${FEATURE_SRC} ${MESH_SRC} ${STITCHING_SRC} ${UTIL_SRC} Feature/vlfeat_sift.cpp Util/StitchOrder.h Util/SurfCuda.h)
target_link_libraries(NISwGSP ${OpenCV_LIBS} )  #${VLFEAT_LIBRARY}
