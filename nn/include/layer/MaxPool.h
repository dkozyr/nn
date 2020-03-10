#pragma once

#include "layer/Layer.h"

namespace nn {
namespace layer {

template <ExecutorType T = Cpu>
class MaxPool : public layer::Layer<T> {
public:
    explicit MaxPool(const Shape& input, const Shape& window, const size_t stride, const std::string& name = "")
        : layer::Layer<T>(input.Size(), GetOutputNeurons(input, window, stride), name)
        , _input_shape(input)
        , _window_shape(window)
        , _stride(stride)
        , _dFdY_offset(Shape{kMaxBatch, GetOutputNeurons(input, window, stride)}) {
    }

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        if((this->_dFdX.GetShape().cols != shape.cols) || (shape.rows > kMaxBatch)) {
            throw "MaxPool forward: wrong matrix shape";
        }

        const auto N = shape.rows;
        this->_Y.Reshape(Shape{N, this->_out_neurons});
        this->_dFdX.Reshape(shape);
        this->_dFdY_offset.SetZeroValue();

        const auto max_row = _input_shape.rows - _window_shape.rows;
        const auto max_col = _input_shape.cols - _window_shape.cols;

        for(size_t n = 0, x_offset = 0, i = 0; n < N; ++n) {
            for(size_t layer = 0; layer < _input_shape.layers; ++layer) {
                const uint32_t layer_offset = x_offset + layer * _input_shape.LayerSize();
                for(size_t row = 0; row  <= max_row; row += _stride) {
                    for(size_t col = 0; col <= max_col; col += _stride, ++i) {
                        const uint32_t pos_offset = layer_offset + row * _input_shape.RowSize() + col;
                        float value = X[pos_offset];
                        uint32_t best_pos = pos_offset;

                        auto offset = pos_offset;
                        for(size_t j = 0; (j < _window_shape.rows) && (row + j < _input_shape.rows); ++j) {
                            for(size_t k = 0; (k < _window_shape.cols) && (col + k < _input_shape.cols); ++k) {
                                if(X[offset + k] > value) {
                                    value = X[offset + k];
                                    best_pos = offset + k;
                                }
                            }
                            offset += _input_shape.RowSize();
                        }

                        this->_Y[i] = value;
                        this->_dFdY_offset[i] = best_pos;
                    }
                }
            }
            x_offset += _input_shape.Size();
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float /*learning_rate*/) override final {
        const auto& shape = X.GetShape();
        if((this->_dFdX.GetShape().cols != shape.cols) || (shape.rows > kMaxBatch)) {
            throw "MaxPool backprop: wrong matrix shape";
        }

        this->_dFdX.SetZeroValue();

        const auto size = dFdY.GetShape().Size();
        for(size_t i = 0; i < size; ++i) {
            const auto offset = _dFdY_offset[i];
            this->_dFdX[offset] += dFdY[i];
        }
    }

    static /*constexpr*/ size_t GetOutputNeurons(const Shape& input, const Shape& window, const size_t stride) {
        const auto rows = CalculateNeurons(input.rows, window.rows, stride);
        const auto cols = CalculateNeurons(input.cols, window.cols, stride);
        return input.layers * rows * cols;
    }

private:
    static /*constexpr*/ size_t CalculateNeurons(const size_t size, const size_t window, const size_t stride) {
        if((size <= window) || (size <= stride)) {
            return 1;
        } else {
            return 1 + CalculateNeurons(size - stride, window, stride);
        }
    }

private:
    const Shape _input_shape;
    const Shape _window_shape;
    const size_t _stride;

    Matrix<T, uint32_t> _dFdY_offset;
};

} //namespace layer
} //namespace nn
