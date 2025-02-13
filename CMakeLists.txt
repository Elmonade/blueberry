cmake_minimum_required(VERSION 3.30)
project(blueberry)

set(CMAKE_CXX_STANDARD 14)

add_executable(blueberry
    matrixMult/multiThread.cpp
    matrixMult/read.cpp
    matrixMult/simple.cpp
    matrixMult/singleThread.cpp
    matrixMult/read.h)
