#if USE_CUDA

#include "cuda/Common.h"
#include "layer/Conv3D.h"

namespace nn {
namespace layer {

__global__ void Conv3DForwardImpl(const int N, const int in_layers, const int in_rows, const int in_cols,
                                  const int out_layers, const int out_rows, const int out_cols,
                                  const int conv_rows, const int conv_cols,
                                  const float* X, const float* W, float* Y) {
    const int n_and_out_layer = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_layer = n_and_out_layer / N;
    const int n = n_and_out_layer - out_layer * N;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;

    if((n < N) && (out_layer < out_layers) && (row < out_rows) && (col < out_cols)) {
        const int y_offset = ((n * out_layers + out_layer) * out_rows + row) * out_cols + col;
        const int w_offset = out_layer * in_layers * conv_rows * conv_cols;

        float value = 0;
        int x_offset = (n * in_layers * in_rows + row) * in_cols + col;
        for(size_t in_layer = 0, w = 0; in_layer < in_layers; ++in_layer, x_offset += in_rows * in_cols) {
            for(size_t j = 0; j < conv_rows; ++j) {
                int xpos = x_offset + j * in_cols;
                for(size_t i = 0; i < conv_cols; ++i, ++w, ++xpos) {
                    value += W[w_offset + w] * X[xpos];
                }
            }
        }
        Y[y_offset] = value;
    }
}

__global__ void Conv3DBackpropImpl(const int N, const int in_layers, const int in_rows, const int in_cols,
                                   const int out_layers, const int out_rows, const int out_cols,
                                   const int conv_rows, const int conv_cols,
                                   const float* X, const float* W, const float* dFdY, float* dFdX, float* dFdW) {
    const int n_and_out_layer = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_layer = n_and_out_layer / N;
    const int n = n_and_out_layer - out_layer * N;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.z * blockDim.z + threadIdx.z;
                                
    if((n < N) && (out_layer < out_layers) && (row < out_rows) && (col < out_cols)) {
        const int y_offset = ((n * out_layers + out_layer) * out_rows + row) * out_cols + col;
        const auto dfdy = dFdY[y_offset];
        const int w_offset = out_layer * in_layers * conv_rows * conv_cols;

        int x_offset = (n * in_layers * in_rows + row) * in_cols + col;
        for(size_t in_layer = 0, w = 0; in_layer < in_layers; ++in_layer, x_offset += in_rows * in_cols) {
            for(size_t j = 0; j < conv_rows; ++j) {
                int xpos = x_offset + j * in_cols;
                for(size_t i = 0; i < conv_cols; ++i, ++w, ++xpos) {
                    dFdX[xpos] += W[w_offset + w] * dfdy;
                    dFdW[w_offset + w] += X[xpos] * dfdy;
                }
            }
        }
    }
}

__global__ void Conv3DUpdateWeightsImpl(const int N, const float* dFdW, float* W, float delta) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < N) {
        W[n] -= delta * dFdW[n];
    }
}

template <>
void Conv3D<Cuda>::Forward(const Matrix<Cuda>& X) {
    const auto& shape = X.GetShape();
    if(this->_in_neurons != shape.cols) {
        throw "Conv3D forward: wrong matrix shape";
    }

    const auto& N = shape.rows;
    this->_Y.Reshape(Shape{N, this->_out_neurons});
    this->_Y.SetZeroValue();

    const auto& in_layers = this->_input_shape.layers;
    const auto& in_rows = this->_input_shape.rows;
    const auto& in_cols = this->_input_shape.cols;
    const auto& out_layers = this->_output_shape.layers;
    const auto& out_rows = this->_output_shape.rows;
    const auto& out_cols = this->_output_shape.cols;

    dim3 block_size(8, 8, 8);
    dim3 num_of_blocks((N * out_layers + block_size.x - 1) / block_size.x,
                       (out_rows + block_size.y - 1) / block_size.y,
                       (out_cols + block_size.z - 1) / block_size.z);

    Conv3DForwardImpl<<<num_of_blocks, block_size>>>
        (N, in_layers, in_rows, in_cols, out_layers, out_rows, out_cols,
        this->_conv_shape.rows, this->_conv_shape.cols,
        X.DeviceData(), this->_W.DeviceData(), this->_Y.DeviceData());
    Exception::ThrowOnError("Conv3D: cannot perform forward propagation");
}

template <>
void Conv3D<Cuda>::Backprop(const Matrix<Cuda>& X, const Matrix<Cuda>& dFdY, float learning_rate) {
    const auto& x_shape = X.GetShape();
    const auto& y_shape = dFdY.GetShape();
    const auto& N = x_shape.rows;
    if((y_shape.cols != this->_out_neurons) || (y_shape.rows != N) || (this->_in_neurons != x_shape.cols)) {
        throw "Conv3D backprop: wrong matrix shape";
    }

    this->_dFdX.Reshape(x_shape);
    this->_dFdX.SetZeroValue();
    this->_dFdW.SetZeroValue();

    const auto& in_layers = this->_input_shape.layers;
    const auto& in_rows = this->_input_shape.rows;
    const auto& in_cols = this->_input_shape.cols;
    const auto& out_layers = this->_output_shape.layers;
    const auto& out_rows = in_rows - this->_conv_shape.rows + 1;
    const auto& out_cols = in_cols - this->_conv_shape.cols + 1;

    {
        dim3 block_size(8, 8, 8);
        dim3 num_of_blocks((N * out_layers + block_size.x - 1) / block_size.x,
                           (out_rows + block_size.y - 1) / block_size.y,
                           (out_cols + block_size.z - 1) / block_size.z);

        Conv3DBackpropImpl<<<num_of_blocks, block_size>>>
            (N, in_layers, in_rows, in_cols, out_layers, out_rows, out_cols,
             this->_conv_shape.rows, this->_conv_shape.cols,
             X.DeviceData(), this->_W.DeviceData(), dFdY.DeviceData(), this->_dFdX.DeviceData(), this->_dFdW.DeviceData());
        Exception::ThrowOnError("Conv3D: cannot perform forward propagation");
    }
    {
        const float delta = learning_rate / N;
        const size_t K = out_layers * this->_conv_shape.Size();
        dim3 block_size(kVectorBlockSize);
        dim3 num_of_blocks((K + block_size.x - 1) / block_size.x);

        Conv3DUpdateWeightsImpl<<<num_of_blocks, block_size>>>
            (K, this->_dFdW.DeviceData(), this->_W.DeviceData(), delta);
        Exception::ThrowOnError("Conv3D: cannot perform back propagation");
    }
}

} //namespace layer
} //namespace nn

#endif //USE_CUDA
