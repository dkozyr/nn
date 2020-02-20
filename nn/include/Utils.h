#pragma once

#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

static random_device g_rd;
static mt19937 g_gen(g_rd());

template<typename T = size_t>
inline T Rand(T a, T b) {
    uniform_int_distribution<T> dis(a, b);
    return dis(g_gen);
}

template<>
inline float Rand(float a, float b) {
    uniform_real_distribution<float> dis(a, b);
    return dis(g_gen);
}

inline float RandNormal(float mu, float sigma) {
    normal_distribution<float> dis(mu, sigma);
    return dis(g_gen);
}

inline std::vector<size_t> GetShuffledIndex(size_t num) {
    std::vector<size_t> I(num);
    for(size_t i = 0; i < num; ++i) {
        I[i] = i;
    }
    std::shuffle(I.begin(), I.end(), g_gen);
    return I;
}

inline uint32_t GetHashString(const string& s) {
    uint32_t hash = 0;
    for(auto& c: s) {
        hash += c;
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

inline uint32_t ReadUint32(const std::string& str, size_t i) {
    size_t result = (uint8_t)str[i];
    result = (result << 8) + (uint8_t)str[i + 1];
    result = (result << 8) + (uint8_t)str[i + 2];
    result = (result << 8) + (uint8_t)str[i + 3];
    return result;
}
