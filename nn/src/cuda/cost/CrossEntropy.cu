#if USE_CUDA

#include "Common.h"
#include "cuda/Common.h"
#include "cost/CrossEntropy.h"

namespace nn {
namespace cost {

__global__ void CrossEntropyWithDerivativeImpl(const int N, const float* Yh, const float* Y, float* dY, float* cost) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        dY[i] = Yh[i] - Y[i];
        if(Y[i] > 0) {
            float partial_cost;
            if(Yh[i] > 0) {
                partial_cost = Y[i] * logf(Yh[i]);
            } else {
                partial_cost = -1e3;
            }
            atomicAdd(cost, -partial_cost);
        }
    }
}

template <>
void CrossEntropy<Cuda>::Evaluate(const Matrix<Cuda>& Yh, const Matrix<Cuda>& Y) {
    const auto& shape = Yh.GetShape();
    const auto N = shape.Size();
    if((shape.rows != Y.GetShape().rows) || (Y.GetShape().cols != shape.cols)) {
        throw "CrossEntropy Evaluate: wrong shape size";
    }
    this->_dFdX.Reshape(shape);

    float* cost;
    cudaMallocManaged(&cost, sizeof(float));
    *cost = 0.0f;

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    CrossEntropyWithDerivativeImpl<<<num_of_blocks, block_size>>>
        (N, Yh.DeviceData(), Y.DeviceData(), this->_dFdX.DeviceData(), cost);
    cudaDeviceSynchronize();
    Exception::ThrowOnError("CrossEntropy: cannot compute derivative for binary cross entropy");

    this->_loss.samples += shape.rows;
    this->_loss.value += *cost;
    cudaFree(cost);
}

} //namespace cost
} //namespace nn

#endif //USE_CUDA
