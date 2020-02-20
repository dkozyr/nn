#pragma once

#include <exception>
#include <iostream>

namespace nn {

#if USE_CUDA

constexpr auto kCudaMaxRandomStates = 256;
constexpr auto kVectorBlockSize = 256;
constexpr auto kMatrixBlockSize = 16;

class Exception : std::exception {
public:
    Exception(const char* message)
        : _message(message) {
        std::cout << "CUDA exception: " << _message << std::endl;
    }

    virtual const char* what() const throw() {
        return _message;
    }

    static void ThrowOnError(const char* message) {
        const cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess) {
            std::cerr << error << ": " << message << std::endl;
            throw Exception(message);
        }
    }

private:
    const char* _message;
};

#endif // USE_CUDA

} // namespace nn
