#if USE_CUDA

#include "cuda/Common.h"
#include "layer/Conv.h"

namespace nn {
namespace layer {

__global__ void ConvForwardImpl(const int N, const int layers, const int in_rows, const int in_cols,
                                const int out_rows, const int out_cols, const int conv_rows, const int conv_cols,
                                const float* X, const float* W, float* Y) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int layer_and_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int layer = layer_and_row / out_rows;
    const int row = layer_and_row - layer * out_rows;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;

    if((n < N) && (layer < layers) && (row < out_rows) && (col < out_cols)) {
        const int x_offset = (n * layers + layer) * in_rows * in_cols + row * in_cols + col;
        const int y_offset = (n * layers + layer) * out_rows * out_cols + row * out_cols + col;

        float value = 0;
        for(size_t j = 0; j < conv_rows; ++j) {
            for(size_t i = 0; i < conv_cols; ++i) {
                const int w = j * conv_cols + i;
                value += W[w] * X[x_offset + j*in_cols + i];
            }
        }
        Y[y_offset] = value;
    }
}

__global__ void ConvBackpropImpl(const int N, const int layers, const int in_rows, const int in_cols,
                                 const int out_rows, const int out_cols, const int conv_rows, const int conv_cols,
                                 const float* X, const float* W, const float* dFdY, float* dFdX, float* dFdW) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int layer_and_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int layer = layer_and_row / out_rows;
    const int row = layer_and_row - layer * out_rows;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;

    if((n < N) && (layer < layers) && (row < out_rows) && (col < out_cols)) {
        const int x_offset = (n * layers + layer) * in_rows * in_cols + row * in_cols + col;
        const int y_offset = (n * layers + layer) * out_rows * out_cols + row * out_cols + col;
        const auto dfdy = dFdY[y_offset];

        for(size_t j = 0; j < conv_rows; ++j) {
            for(size_t i = 0; i < conv_cols; ++i) {
                const int w = j * conv_cols + i;
                dFdX[x_offset + j*in_cols + i] += W[w] * dfdy;
                dFdW[w] += X[x_offset + j*in_cols + i] * dfdy;
            }
        }
    }
}

__global__ void ConvUpdateWeightsImpl(const int N, const int conv_rows, const int conv_cols,
                                      const float* dFdW, float* W, float learning_rate) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if((row < conv_rows) && (col < conv_cols)) {
        const int i = row * conv_cols + col;
        W[i] -= learning_rate * dFdW[i] / N;
    }
}

template <>
void Conv<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if(this->_in_neurons != shape.cols) {
        throw "Conv forward: wrong matrix shape";
    }

    const auto& N = shape.rows;
    this->_Y.Reshape(Shape{N, this->_out_neurons});
    this->_Y.SetZeroValue();

    const auto& layers = this->_input_shape.layers;
    const auto& in_rows = this->_input_shape.rows;
    const auto& in_cols = this->_input_shape.cols;
    const auto& out_rows = in_rows - this->_conv_shape.rows + 1;
    const auto& out_cols = in_cols - this->_conv_shape.cols + 1;

    dim3 block_size(8, 8, 8);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x,
                       (layers * out_rows + block_size.y - 1) / block_size.y,
                       (out_cols + block_size.z - 1) / block_size.z);

    ConvForwardImpl<<<num_of_blocks, block_size>>>
        (N, layers, in_rows, in_cols, out_rows, out_cols, this->_conv_shape.rows, this->_conv_shape.cols,
        X.DeviceData(), this->_W.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("Conv: cannot perform forward propagation");
}

template <>
void Conv<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, float learning_rate) {
    const auto& x_shape = X.GetShape();
    const auto& y_shape = dFdY.GetShape();
    const auto& N = x_shape.rows;
    if((y_shape.cols != this->_out_neurons) || (y_shape.rows != N) || (this->_in_neurons != x_shape.cols)) {
        throw "Conv backprop: wrong matrix shape";
    }

    this->_dFdX.Reshape(x_shape);
    this->_dFdX.SetZeroValue();
    this->_dFdW.SetZeroValue();

    const auto& layers = this->_input_shape.layers;
    const auto& in_rows = this->_input_shape.rows;
    const auto& in_cols = this->_input_shape.cols;
    const auto& out_rows = in_rows - this->_conv_shape.rows + 1;
    const auto& out_cols = in_cols - this->_conv_shape.cols + 1;

    dim3 block_size(8, 8, 8);
    dim3 num_of_blocks((N + block_size.x - 1) / block_size.x,
                       (layers * out_rows + block_size.y - 1) / block_size.y,
                       (out_cols + block_size.z - 1) / block_size.z);

    ConvBackpropImpl<<<num_of_blocks, block_size>>>
        (N, layers, in_rows, in_cols, out_rows, out_cols, this->_conv_shape.rows, this->_conv_shape.cols,
        X.DeviceData(), this->_W.DeviceData(), dFdY.DeviceData(), this->_dFdX.DeviceData(), this->_dFdW.DeviceData());
    Exception::ThrowOnError("Conv: cannot perform forward propagation");

    {
        dim3 block_size(kMatrixBlockSize, kMatrixBlockSize);
        dim3 num_of_blocks((this->_conv_shape.rows + block_size.x - 1) / block_size.x,
                           (this->_conv_shape.cols + block_size.y - 1) / block_size.y);

        ConvUpdateWeightsImpl<<<num_of_blocks, block_size>>>
            (N, this->_conv_shape.rows, this->_conv_shape.cols,
            this->_dFdW.DeviceData(), this->_W.DeviceData(), learning_rate);
        Exception::ThrowOnError("Conv: cannot perform back propagation");
    }
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
