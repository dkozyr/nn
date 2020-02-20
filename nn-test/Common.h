#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <type_traits>

#include "Clock.h"
#include "Utils.h"

namespace nn {

constexpr float kEps = 1e-2;

inline static void AssertEqual(float a, float b) {
    if((std::abs(a) < kEps) || (std::abs(b) < kEps)) {
        EXPECT_NEAR(a, b, kEps);
    } else {
        EXPECT_NEAR(a / b - 1.0, 0.0, kEps);
    }
}

} //namespace nn