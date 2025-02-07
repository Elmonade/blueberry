cmake_minimum_required(VERSION 3.30)
project(blueberry)

set(CMAKE_CXX_STANDARD 14)

add_executable(blueberry
    multiThread.cpp
    read.cpp
    simple.cpp
    singleThread.cpp)
