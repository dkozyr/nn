#if USE_CUDA

#include "cuda/Common.h"
#include "activation/Softmax.h"

namespace nn {
namespace activation {

__global__ void SoftmaxForwardImpl(const int N, const int neurons, const float* X, float* Y) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N) {
        int offset = n * neurons;

        float max_element = X[offset];
        for(int i = 1; i < neurons; ++i) {
            if(X[offset + i] > max_element) {
                max_element = X[offset + i];
            }
        }

        float sum = 0;
        for(int i = 0; i < neurons; ++i) {
            Y[offset + i] = exp(X[offset + i] - max_element);
            sum += Y[offset + i];
        }

        float factor = 1.0f / sum;
        for(int i = 0; i < neurons; ++i) {
            Y[offset + i] *= factor;
        }
    }
}

template <>
void Softmax<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_Y.GetShape().cols != shape.cols)) {
        throw Exception("Softmax forward: wrong matrix shape");
    }

    const auto N = shape.rows;
    this->_Y.Reshape(Shape{N, this->_out_neurons});
    this->_dFdX.Reshape(Shape{N, this->_in_neurons});

    dim3 block_size(kVectorBlockSize);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x);

    SoftmaxForwardImpl<<<num_of_blocks, block_size>>>
        (N, this->_out_neurons, X.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("Softmax: cannot perform forward propagation");
}

template <>
void Softmax<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, const float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
       (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
        throw Exception("Softmax backprop: wrong matrix shape");
    }

    const auto N = shape.Size();
    cudaMemcpy(this->_dFdX.DeviceData(), dFdY.DeviceData(), N * sizeof(float), cudaMemcpyDeviceToDevice);
    Exception::ThrowOnError("Softmax: cannot perform back propagation");
}

} //namespace activation
} //namespace nn

#endif //USE_CUDA
