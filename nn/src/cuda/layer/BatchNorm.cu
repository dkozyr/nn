#if USE_CUDA

#include "cuda/Common.h"
#include "layer/BatchNorm.h"

namespace nn {
namespace layer {

__global__ void BatchNormForwardImpl(const int N, const int neurons, const float* X, const float* mean, const float* sigma,
                                     const float* gamma, const float* beta, float* xhat, float* Y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < neurons) {
        for(int n = 0, offset = i; n < N; ++n, offset += neurons) {
            xhat[offset] = (X[offset] - mean[i]) / (sigma[i] + 0.001f);
            Y[offset] = gamma[i] * xhat[offset] + beta[i];
        }
    }
}

__global__ void BatchNormBackpropImpl(const int N, const int neurons, const float* xhat, const float* mean, const float* sigma,
                                      float* gamma, float* beta, const float* dFdY, float* dFdX, float learning_rate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < neurons) {
        float dFdGamma = 0.0f, dFdBeta = 0.0f;
        for(int n = 0, offset = i; n < N; ++n, offset += neurons) {
            dFdGamma += dFdY[offset] * xhat[offset];
            dFdBeta += dFdY[offset];
        }

        const float factor = gamma[i] * sigma[i] / N;
        for(int n = 0, offset = i; n < N; ++n, offset += neurons) {
            dFdX[offset] = factor * (N * dFdY[offset] - dFdGamma * xhat[offset] - dFdBeta);
        }

        const float delta = learning_rate / N;
        gamma[i] -= delta * dFdGamma;
        beta[i] -= delta * dFdBeta;
    }
}

__global__ void CalculateMeanImpl(const int N, const int neurons, const float* X, float* mean) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < neurons) {
        float m = 0.0f;
        for(int n = 0, offset = i; n < N; ++n, offset += neurons) {
            m += X[offset];
        }
        mean[i] = m / N;
    }
}

__global__ void CalculateStdDevImpl(const int N, const int neurons, const float* X, const float* mean, float* sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < neurons) {
        float sigma2 = 0.0f;
        for(int n = 0, offset = i; n < N; ++n, offset += neurons) {
            const auto x_zero_mean = X[offset] - mean[i];
            sigma2 += x_zero_mean * x_zero_mean;
        }
        sigma[i] = sqrt(sigma2 / N);
    }
}

template <>
void BatchNorm<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("BatchNorm forward: wrong matrix shape");
    }

    const auto& N = shape.rows;
    const auto& neurons = shape.cols;
    this->_Y.Reshape(shape);
    this->_dFdX.Reshape(shape);
    this->_xhat.Reshape(shape);

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((neurons + block_size.x - 1) / block_size.x);

    CalculateMeanImpl<<<num_of_blocks, block_size>>>
        (N, neurons, X.DeviceData(), this->_mean.DeviceData());
    Exception::ThrowOnError("BatchNorm: cannot calculate Mean values");

    CalculateStdDevImpl<<<num_of_blocks, block_size>>>
        (N, neurons, X.DeviceData(), this->_mean.DeviceData(), this->_sigma.DeviceData());
    Exception::ThrowOnError("BatchNorm: cannot calculate StdDev values");

    BatchNormForwardImpl<<<num_of_blocks, block_size>>>
        (N, neurons, X.DeviceData(), this->_mean.DeviceData(), this->_sigma.DeviceData(),
         this->_gamma.DeviceData(), this->_beta.DeviceData(), this->_xhat.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("BatchNorm: cannot perform forward propagation");
}

template <>
void BatchNorm<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, float learning_rate) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
        throw Exception("BatchNorm backprop: wrong matrix shape");
    }

    const auto& N = shape.rows;
    const auto& neurons = shape.cols;
    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((neurons + block_size.x - 1) / block_size.x);

    BatchNormBackpropImpl<<<num_of_blocks, block_size>>>
        (N, neurons, this->_xhat.DeviceData(), this->_mean.DeviceData(), this->_sigma.DeviceData(),
        this->_gamma.DeviceData(), this->_beta.DeviceData(), dFdY.DeviceData(), this->_dFdX.DeviceData(), learning_rate);
    Exception::ThrowOnError("BatchNorm: cannot perform back propagation");
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
