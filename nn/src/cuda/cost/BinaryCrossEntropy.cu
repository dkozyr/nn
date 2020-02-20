#if USE_CUDA

#include "cuda/Common.h"
#include "cost/BinaryCrossEntropy.h"

namespace nn {
namespace cost {

__global__ void BinaryCrossEntropyImpl(const int N, const float* Yh, const float* Y, float* cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        float partial_cost;
        if(Yh[i] == Y[i]) {
            partial_cost = 0;
        } if((Yh[i] == 0) || (Yh[i] == 1.0f)) {
            partial_cost = -10;
        } else {
            partial_cost = Y[i] * logf(Yh[i]) + (1.0f - Y[i]) * logf(1.0f - Yh[i]);
        }
        atomicAdd(cost, -partial_cost);
    }
}

__global__ void BinaryCrossEntropyDerivativeImpl(const int N, const float* Yh, const float* Y, float* dY) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        if(Yh[i] == Y[i]) {
            dY[i] = 0;
        } else if(Yh[i] == 0) {
            dY[i] = +10.0f;
        } else if(Yh[i] == 1.0f) {
            dY[i] = -10.0f;
        } else {
            dY[i] = -(Y[i] / Yh[i] - (1.0f - Y[i]) / (1.0f - Yh[i]));
        }
    }
}

template <>
void BinaryCrossEntropy<Cuda>::Evaluate(const Matrix<Cuda>& Yh, const Matrix<Cuda>& Y) {
    const auto& shape = Yh.GetShape();
    const auto N = shape.rows;
    if((shape.rows != Y.GetShape().rows) || (Y.GetShape().cols != 1) || (shape.cols != 1)) {
        throw "BinaryCrossEntropy Evaluate: wrong shape size";
    }
    this->_dFdX.Reshape(shape);

    float* cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    BinaryCrossEntropyDerivativeImpl<<<num_of_blocks, block_size>>>
        (N, Yh.DeviceData(), Y.DeviceData(), this->_dFdX.DeviceData());
    cudaDeviceSynchronize();
    Exception::ThrowOnError("Cannot compute derivative for binary cross entropy");

    BinaryCrossEntropyImpl<<<num_of_blocks, block_size>>>
        (N, Yh.DeviceData(), Y.DeviceData(), cost);
    cudaDeviceSynchronize();
    Exception::ThrowOnError("Cannot compute binary cross entropy cost");

    this->_loss.samples += shape.rows;
    this->_loss.value += *cost;
    cudaFree(cost);
}

} //namespace cost
} //namespace nn

#endif //USE_CUDA
