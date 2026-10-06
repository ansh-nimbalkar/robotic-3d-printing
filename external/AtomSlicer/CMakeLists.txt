cmake_minimum_required(VERSION 3.10...4.1)
project(slicer)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE sources      src/*.cpp src/*.h)

add_executable(slicer ${sources})
target_compile_options(slicer PUBLIC -Wall -O3 -march=native)