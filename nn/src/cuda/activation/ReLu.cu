#if USE_CUDA

#include "cuda/Common.h"
#include "activation/ReLu.h"

namespace nn {
namespace activation {

__global__ void ReLuForwardImpl(const int N, const float* X, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        Y[i] = fmaxf(X[i], 0);
    }
}

__global__ void ReLuBackpropImpl(const int N, const float* dFdY, const float* X, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        if(X[i] > 0) {
            dFdX[i] = dFdY[i];
        } else {
            dFdX[i] = 0.0;
        }
    }
}

template <>
void ReLu<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("ReLu forward: wrong matrix shape");
    }

    const auto N = shape.Size();
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    ReLuForwardImpl<<<num_of_blocks, block_size>>>
        (N, X.DeviceData(), this->_Y.DeviceData());
	Exception::ThrowOnError("ReLu: cannot perform forward propagation");
}

template <>
void ReLu<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows != this->_Y.GetShape().rows)) {
        throw Exception("ReLu backprop: wrong matrix shape");
    }

    const auto N = shape.Size();
    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    ReLuBackpropImpl<<<num_of_blocks, block_size>>>
        (N, dFdY.DeviceData(), X.DeviceData(), this->_dFdX.DeviceData());
    Exception::ThrowOnError("ReLu: cannot perform back propagation");
}

} //namespace activation
} //namespace nn

#endif //USE_CUDA
