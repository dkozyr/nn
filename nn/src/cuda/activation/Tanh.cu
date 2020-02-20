#if USE_CUDA

#include "cuda/Common.h"
#include "activation/Tanh.h"

namespace nn {
namespace activation {

__global__ void TanhForwardImpl(const int N, const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        Y[i] = tanh(X[i]);
    }
}

__global__ void TanhBackpropImpl(const int N, const float* dFdY, const float* Y, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        dFdX[i] = dFdY[i] * (1.0f - Y[i] * Y[i]);
    }
}

template <>
void Tanh<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("Tanh forward: wrong matrix shape");
    }

    const auto N = shape.Size();
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    TanhForwardImpl<<<num_of_blocks, block_size>>>
        (N, X.DeviceData(), this->_Y.DeviceData());
	Exception::ThrowOnError("Tanh: cannot perform forward propagation");
}

template <>
void Tanh<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
        throw Exception("Tanh backprop: wrong matrix shape");
    }

    const size_t N = shape.Size();

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    TanhBackpropImpl<<<num_of_blocks, block_size>>>
        (N, dFdY.DeviceData(), this->_Y.DeviceData(), this->_dFdX.DeviceData());
    Exception::ThrowOnError("Tanh: cannot perform back propagation");
}

} //namespace activation
} //namespace nn

#endif //USE_CUDA
