#pragma once

#include <algorithm>
#include <exception>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>

namespace nn {

enum ExecutorType {
    Cpu = 0,
    Cuda = 1
};

constexpr auto kMaxBatch = 4096;

inline constexpr bool IsCudaEnabled() {
#if USE_CUDA
    return true;
#else
    return false;
#endif
}

} //namespace nn
