#if USE_CUDA

#include "cuda/Common.h"
#include "activation/Sigmoid.h"

namespace nn {
namespace activation {

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__global__ void SigmoidForwardImpl(const int N, const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        Y[i] = sigmoid(X[i]);
    }
}

__global__ void SigmoidBackpropImpl(const int N, const float* dFdY, const float* Y, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        dFdX[i] = dFdY[i] * Y[i] * (1.0f - Y[i]);
    }
}

template <>
void Sigmoid<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("Sigmoid forward: wrong matrix shape");
    }

    const auto N = shape.Size();
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    SigmoidForwardImpl<<<num_of_blocks, block_size>>>
        (N, X.DeviceData(), this->_Y.DeviceData());
	Exception::ThrowOnError("Sigmoid: cannot perform forward propagation");
}

template <>
void Sigmoid<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows != this->_Y.GetShape().rows)) {
        throw Exception("Sigmoid backprop: wrong matrix shape");
    }

    const auto N = shape.Size();
    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    SigmoidBackpropImpl<<<num_of_blocks, block_size>>>
        (N, dFdY.DeviceData(), this->_Y.DeviceData(), this->_dFdX.DeviceData());
    Exception::ThrowOnError("Sigmoid: cannot perform back propagation");
}

} //namespace activation
} //namespace nn

#endif //USE_CUDA
