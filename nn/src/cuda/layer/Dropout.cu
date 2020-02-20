#if USE_CUDA

// TODO: fix random generator
// #include <curand.h>
// #include <curand_kernel.h>

#include "cuda/Common.h"
#include "layer/Dropout.h"

namespace nn {
namespace layer {

__global__ void RandomsInit(uint32_t seed, uint32_t* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    seed = seed * ((unsigned)(i + 1) * 13u);
    states[i] = seed;
}

__device__ float generate(uint32_t* state, int i) {
    uint32_t x = state[i] * 16843009u + 826366247u;
    state[i] = x;
    return (float)x / (float)0x100000000;
}

__global__ void DropoutForwardImpl(const int N, const float* X, const float probability, uint32_t* states, float* Y, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float rnd = generate(states, threadIdx.x);
        if(rnd < probability) {
            Y[i] = 0;
            dFdX[i] = -1.0;
        } else {
            Y[i] = X[i];
            dFdX[i] = 1.0;
        }
    }
}

__global__ void DropoutBackpropImpl(const int N, const float* dFdY, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        if(dFdX[i] > 0.0) {
            dFdX[i] = dFdY[i];
        } else {
            dFdX[i] = 0.0;
        }
    }
}

template <>
void Dropout<Cuda>::Predict(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("Dropout predict: wrong matrix shape");
    }

    const auto N = shape.Size();
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);

    cudaMemcpy(this->_Y.DeviceData(), X.DeviceData(), N * sizeof(float), cudaMemcpyDeviceToDevice);
	Exception::ThrowOnError("Dropout: cannot perform forward propagation");
}

template <>
void Dropout<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("Dropout forward: wrong matrix shape");
    }

    static std::shared_ptr<uint32_t> states_ptr = nullptr;
    if(!states_ptr) {
        uint32_t* rand_states = nullptr;
        cudaMalloc(&rand_states, kCudaMaxRandomStates * sizeof(uint32_t));
        Exception::ThrowOnError("Dropout: Cannot allocate CUDA memory for curand states");

        states_ptr = std::shared_ptr<uint32_t>(rand_states, [&](uint32_t* ptr){ cudaFree(ptr); });
        RandomsInit<<<1, kCudaMaxRandomStates>>>(time(0), states_ptr.get());
    }

    const auto N = shape.Size();
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);

    dim3 block_size(kCudaMaxRandomStates);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    DropoutForwardImpl<<<num_of_blocks, block_size>>>
        (N, X.DeviceData(), _probability, states_ptr.get(), this->_Y.DeviceData(), this->_dFdX.DeviceData());
    cudaDeviceSynchronize();
    Exception::ThrowOnError("Dropout: cannot perform forward propagation");
}

template <>
void Dropout<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
        throw Exception("Dropout backprop: wrong matrix shape");
    }

    const auto N = shape.Size();
    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    DropoutBackpropImpl<<<num_of_blocks, block_size>>>
        (N, dFdY.DeviceData(), this->_dFdX.DeviceData());
    Exception::ThrowOnError("Dropout: cannot perform back propagation");
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
