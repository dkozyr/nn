FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl

RUN mkdir temp

# CMake
RUN cd temp \
 && wget https://github.com/Kitware/CMake/releases/download/v3.16.3/cmake-3.16.3.tar.gz \
 && tar -xvzf cmake-3.16.3.tar.gz \
 && cd cmake-3.16.3 \
 && ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF \
 && make -j 4 \
 && make install

# GTest
RUN cd temp \
 && wget https://github.com/google/googletest/archive/release-1.10.0.tar.gz \
 && tar -xvzf release-1.10.0.tar.gz \
 && cd googletest-release-1.10.0 \
 && cmake . \
 && make -j 4 \
 && make install

WORKDIR /
