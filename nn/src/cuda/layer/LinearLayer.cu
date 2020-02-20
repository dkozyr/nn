#if USE_CUDA

#include "cuda/Common.h"
#include "layer/LinearLayer.h"

namespace nn {
namespace layer {

__global__ void LinearLayerForwardImpl(const int N, const int in_neurons, const int out_neurons,
                                       const float* W, const float* B, const float* X, float* Y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if((row < N) && (col < out_neurons)) {
        float value = B[col];
        int x_offset = row * in_neurons;
        int w = col;
        for(int i = 0; i < in_neurons; ++i, ++x_offset, w += out_neurons) {
            value += X[x_offset] * W[w];
        }
        Y[row * out_neurons + col] = value;
    }
}

__global__ void LinearLayerBackpropImpl(const int N, const int in_neurons, const int out_neurons,
                                        const float* dFdY, const float* W, float* dFdX) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if((row < N) && (col < in_neurons)) {
        float dfdx = 0;
        int y_offset = row * out_neurons;
        int w = col * out_neurons;
        for(int i = 0; i < out_neurons; ++i, ++y_offset, ++w) {
            dfdx += dFdY[y_offset] * W[w];
        }
        dFdX[row * in_neurons + col] = dfdx;
    }
}

__global__ void LinearLayerUpdateWeightsImpl(const int N, const int in_neurons, const int out_neurons,
                                             const float* dFdY, const float* X, float* W, float delta) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if((row < in_neurons) && (col < out_neurons)) {
        float dfdw = 0;
        int x_offset = row;
        int y_offset = col;
        for(int n = 0; n < N; ++n, x_offset += in_neurons, y_offset += out_neurons) {
            dfdw += dFdY[y_offset] * X[x_offset];
        }
        W[row * out_neurons + col] -= delta * dfdw;
    }
}

__global__ void LinearLayerUpdateBiasImpl(const int N, const int out_neurons,
                                          const float* dFdY, float* B, float delta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < out_neurons) {
        float dfdb = 0;
        int y_offset = col;
        for(int n = 0; n < N; ++n, y_offset += out_neurons) {
            dfdb += dFdY[y_offset];
        }
        B[col] -= delta * dfdb;
    }
}

template <>
void LinearLayer<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_W.GetShape().rows != shape.cols)) {
        throw Exception("LinearLayer forward: wrong matrix shape");
    }

    const auto& N = shape.rows;
    const auto& in_neurons = this->_in_neurons;
    const auto& out_neurons = this->_out_neurons;

    this->_Y.Reshape(Shape{N, out_neurons});
    this->_dFdX.Reshape(Shape{N, in_neurons});

    dim3 block_size(kMatrixBlockSize, kMatrixBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x,
                       (out_neurons + block_size.y - 1) / block_size.y);

    LinearLayerForwardImpl<<<num_of_blocks, block_size>>>
        (N, in_neurons, out_neurons,
        this->_W.DeviceData(), this->_B.DeviceData(), X.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("LinearLayer: cannot perform forward propagation");
}

template <>
void LinearLayer<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float learning_rate) {
    const auto N = X.GetShape().rows;
    const auto& shape = this->_W.GetShape();
    const auto& in_neurons = this->_in_neurons;
    const auto& out_neurons = this->_out_neurons;
    if((dFdY.GetShape().cols != out_neurons) || (X.GetShape().rows != dFdY.GetShape().rows)) {
        throw Exception("LinearLayer backprop: wrong matrix shape");
    }
    
    {
        dim3 block_size(kMatrixBlockSize, kMatrixBlockSize);
        dim3 num_of_blocks((N + block_size.x - 1) / block_size.x,
                           (in_neurons + block_size.y - 1) / block_size.y);

        LinearLayerBackpropImpl<<<num_of_blocks, block_size>>>
            (N, in_neurons, out_neurons,
            dFdY.DeviceData(), this->_W.DeviceData(), this->_dFdX.DeviceData());
        Exception::ThrowOnError("LinearLayer: cannot perform back propagation");
    }

    const float delta = learning_rate / N;
    {
        dim3 block_size(kMatrixBlockSize, kMatrixBlockSize);
        dim3 num_of_blocks((in_neurons + block_size.x - 1) / block_size.x,
                           (out_neurons + block_size.y - 1) / block_size.y);

        LinearLayerUpdateWeightsImpl<<<num_of_blocks, block_size>>>
            (N, in_neurons, out_neurons,
            dFdY.DeviceData(), X.DeviceData(), this->_W.DeviceData(), delta);
        Exception::ThrowOnError("LinearLayer: cannot update weights");
    }
    {
        dim3 block_size(kVectorBlockSize);
        dim3 num_of_blocks((out_neurons + block_size.x - 1) / block_size.x);

        LinearLayerUpdateBiasImpl<<<num_of_blocks, block_size>>>
            (N, out_neurons,
            dFdY.DeviceData(), this->_B.DeviceData(), delta);
        Exception::ThrowOnError("LinearLayer: cannot bias weights");
    }
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
