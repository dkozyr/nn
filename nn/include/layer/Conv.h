#pragma once

#include "Layer.h"

namespace nn {
namespace layer {

// https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199

template <ExecutorType T = Cpu>
class Conv : public Layer<T> {
public:
    explicit Conv(const Shape& input, const Shape& conv, const std::string& name = "")
        : Layer<T>(input.Size(), CalcOutputNeurons(input, conv), name)
        , _input_shape(input)
        , _conv_shape(conv)
        , _W(conv)
        , _dFdW(conv) {
        _W.XavierNormal();
        _W.CopyHostToDevice();
    }

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        if(this->_in_neurons != shape.cols) {
            throw "Conv forward: wrong matrix shape";
        }

        const auto& N = shape.rows;
        this->_Y.Reshape(Shape{N, this->_out_neurons});
        this->_Y.SetZeroValue();

        size_t x_offset = 0, y_offset = 0;
        for(size_t n = 0; n < N; ++n) {
            ConvOperator(X, x_offset, _conv_shape, this->_Y, y_offset);
            x_offset += X.GetShape().RowSize();
            y_offset += this->_Y.GetShape().RowSize();
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, float learning_rate) override final {
        const auto& x_shape = X.GetShape();
        const auto& y_shape = dFdY.GetShape();
        const auto& N = x_shape.rows;
        if((y_shape.cols != this->_out_neurons) || (y_shape.rows != N) || (this->_in_neurons != x_shape.cols)) {
            throw "Conv backprop: wrong matrix shape";
        }

        this->_dFdX.Reshape(x_shape);
        this->_dFdX.SetZeroValue();
        this->_dFdW.SetZeroValue();

        const auto layer_size = _input_shape.LayerSize();
        const auto row_size = _input_shape.RowSize();
        const Shape& conv = _conv_shape;
        const Shape out_shape{_input_shape.rows - _conv_shape.rows + 1, _input_shape.cols - _conv_shape.cols + 1};

        size_t x_offset = 0;
        size_t y_offset = 0;
        for(size_t n = 0; n < N; ++n) {
            auto x_layer_offset = x_offset;
            auto y_layer_offset = y_offset;
            for(size_t layer = 0; layer < _input_shape.layers; ++layer) {
                for(size_t r = 0; r + conv.rows <= _input_shape.rows; ++r) {
                    for(size_t c = 0; c + conv.cols <= _input_shape.cols; ++c) {
                        const auto ypos = y_layer_offset + r*out_shape.RowSize() + c;
                        const auto dfdy = dFdY[ypos];
                        for(size_t j = 0, w = 0; j < conv.rows; ++j) {
                            for(size_t i = 0; i < conv.cols; ++i, ++w) {
                                const auto xpos = x_layer_offset + (r+j)*_input_shape.RowSize() + (c+i);
                                this->_dFdX[xpos] += this->_W[w] * dfdy;
                                this->_dFdW[w] += X[xpos] * dfdy;
                            }
                        }
                    }
                }
                x_layer_offset += _input_shape.LayerSize();
                y_layer_offset += out_shape.LayerSize();
            }
            x_offset += x_shape.RowSize();
            y_offset += y_shape.RowSize();
        }

        const float delta = learning_rate / N;
        for(size_t w = 0; w < conv.rows * conv.cols; ++w) {
            this->_W[w] -= delta * this->_dFdW[w];
        }
    }

    void CopyDeviceToHost() const override final {
        Layer<T>::CopyDeviceToHost();
        _W.CopyDeviceToHost();
    }

    void Save(std::fstream& file) const override final {
        _W.CopyDeviceToHost();
        const auto shape = _W.GetShape();
        file.write(reinterpret_cast<char*>(_W.HostData()), shape.Size() * sizeof(float));
    }

    void Load(std::fstream& file) override final {
        const auto shape = _W.GetShape();
        file.read(reinterpret_cast<char*>(_W.HostData()), shape.Size() * sizeof(float));
        _W.CopyHostToDevice();
    }

    size_t GetOutputNeurons() const {
        return CalcOutputNeurons(_input_shape, _conv_shape);
    }

private:
    void ConvOperator(const Matrix<T>& X, size_t x_offset, const Shape& conv, Matrix<T>& Y, size_t y_offset) {
        const auto layer_size = _input_shape.LayerSize();
        const auto row_size = _input_shape.RowSize();

        for(size_t layer = 0; layer < _input_shape.layers; ++layer) { // RGB
            const auto layer_offset = x_offset + layer * layer_size;

            for(size_t r = 0; r + conv.rows <= _input_shape.rows; ++r) {
                auto conv_offset = layer_offset + r*row_size;
                for(size_t c = 0; c + conv.cols <= _input_shape.cols; ++c) {

                    float value = 0;
                    for(size_t j = 0; j < conv.rows; ++j) {
                        for(size_t i = 0; i < conv.cols; ++i) {
                            value += this->_W(j, i) * X[conv_offset + j*row_size + i];
                        }
                    }
                    Y[y_offset] = value;
                    y_offset++;
                    conv_offset++;
                }
            }
        }
    }

    static size_t CalcOutputNeurons(const Shape input, const Shape conv) {
        return input.layers * (input.rows - conv.rows + 1) * (input.cols - conv.cols + 1);
    }

private:
    const Shape _input_shape;
    const Shape _conv_shape;
    Matrix<T> _W, _dFdW;
};

} //namespace layer
} //namespace nn
