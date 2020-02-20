cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(neural-network LANGUAGES CXX CUDA)

option(USE_CUDA "USE_CUDA" OFF)
if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)

    set(CMAKE_CUDA_FLAGS -arch=sm_50)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
else()
    add_compile_definitions(USE_CUDA=0)
endif()

file(GLOB_RECURSE PROJECT_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.h)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS -pthread)

add_subdirectory(nn)
add_subdirectory(nn-test)
add_subdirectory(example)

# docker run --gpus all -it --rm -v $PWD:/home/dk -w /home/dk nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 nvidia-smi
# docker run --gpus all -it --rm -v $PWD:/home/dk -w /home/dk nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 /bin/bash

# docker run --gpus all -it --rm -v $PWD:/home/dk -w /home/dk cuda-dev /bin/bash

# nn-test/nn-test --gtest_filter=*Mnist*