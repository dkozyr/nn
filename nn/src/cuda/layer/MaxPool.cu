#if USE_CUDA

#include "cuda/Common.h"
#include "layer/MaxPool.h"

namespace nn {
namespace layer {

__global__ void MaxPoolForwardImpl(const int N, const int layers, const int in_rows, const int in_cols,
                                   const int out_rows, const int out_cols,
                                   const int window_rows, const int window_cols, const int stride,
                                   const float* X, uint32_t* dFdY_offset, float* Y) {
    const int n_and_layer = blockIdx.x * blockDim.x + threadIdx.x;
    const int layer = n_and_layer / N;
    const int n = n_and_layer - layer * N;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;

    if((n < N) && (layer < layers) && (row < out_rows) && (col < out_cols)) {
        const uint32_t y_offset = ((n * layers + layer) * out_rows + row) * out_cols + col;
        uint32_t x_offset = ((n * layers + layer) * in_rows + row) * in_cols + col;

        float best_value = X[x_offset];
        uint32_t best_pos = x_offset;

        for(size_t j = 0; (j < window_rows) && (row + j < in_rows); ++j) {
            for(size_t k = 0; (k < window_cols) && (col + k < in_cols); ++k) {
                const float value = X[x_offset + k];
                if(value > best_value) {
                    best_value = value;
                    best_pos = x_offset + k;
                }
            }
            x_offset += in_cols;
        }

        Y[y_offset] = best_value;
        dFdY_offset[y_offset] = best_pos;
    }
}

__global__ void MaxPoolBackpropImpl(const int N, const float* dFdY, const uint32_t* dFdY_offset, float* dFdX) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        const uint32_t offset = dFdY_offset[i];
        dFdX[offset] += dFdY[i];
    }
}

template <>
void MaxPool<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if((this->_dFdX.GetShape().cols != shape.cols) || (shape.rows > kMaxBatch)) {
        Exception::ThrowOnError("MaxPool forward: wrong matrix shape");
    }

    const auto N = shape.rows;
    this->_Y.Reshape(Shape{N, this->_out_neurons});
    this->_dFdX.Reshape(shape);
    this->_dFdY_offset.SetZeroValue();

    const auto& layers = this->_input_shape.layers;
    const auto& in_rows = this->_input_shape.rows;
    const auto& in_cols = this->_input_shape.cols;
    const auto& window_rows = this->_window_shape.rows;
    const auto& window_cols = this->_window_shape.cols;
    const auto& stride = this->_stride;

    const auto out_rows = 1 + ((in_rows - window_rows) + stride - 1) / stride;
    const auto out_cols = 1 + ((in_cols - window_cols) + stride - 1) / stride;

    dim3 block_size(8, 8, 8);
    dim3 num_of_blocks((N * layers + block_size.x - 1) / block_size.x,
                       (out_rows + block_size.y - 1) / block_size.y,
                       (out_cols + block_size.z - 1) / block_size.z);

    MaxPoolForwardImpl<<<num_of_blocks, block_size>>>
        (N, layers, in_rows, in_cols, out_rows, out_cols,
         window_rows, window_cols, stride,
         X.DeviceData(), this->_dFdY_offset.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("MaxPool: cannot perform forward propagation");
}

template <>
void MaxPool<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, float /*learning_rate*/) {
    const auto& shape = X.GetShape();
    if((this->_dFdX.GetShape().cols != shape.cols) || (shape.rows > kMaxBatch)) {
        Exception::ThrowOnError("MaxPool backprop: wrong matrix shape");
    }

    this->_dFdX.SetZeroValue();

    const auto size = dFdY.GetShape().Size();
    dim3 block_size(256);
    dim3 num_of_blocks((size + block_size.x - 1) / block_size.x);

    MaxPoolBackpropImpl<<<num_of_blocks, block_size>>>
        (size, dFdY.DeviceData(), this->_dFdY_offset.DeviceData(), this->_dFdX.DeviceData());
    Exception::ThrowOnError("MaxPool: cannot perform forward propagation");
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
