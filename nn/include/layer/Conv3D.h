#pragma once

#include "Layer.h"
#include "Matrix3D.h"

namespace nn {
namespace layer {

template <ExecutorType T = Cpu>
class Conv3D : public Layer<T> {
public:
    explicit Conv3D(const Shape& input, const Shape& conv, const std::string& name = "")
        : Layer<T>(input.Size(), CalcOutputNeurons(input, conv, conv.layers), name)
        , _input_shape(input)
        , _output_shape{conv.layers, _input_shape.rows - conv.rows + 1, _input_shape.cols - conv.cols + 1}
        , _conv_shape(input.layers, conv.rows, conv.cols)
        , _W(Shape{conv.layers * _conv_shape.layers, _conv_shape.rows, _conv_shape.cols})
        , _dFdW(Shape{conv.layers * _conv_shape.layers, _conv_shape.rows, _conv_shape.cols}) {
        _W.XavierNormal();
        _W.CopyHostToDevice();
    }

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        if(this->_in_neurons != shape.cols) {
            throw "Conv3D forward: wrong matrix shape";
        }

        const auto& N = shape.rows;
        this->_Y.Reshape(Shape{N, this->_out_neurons});
        this->_Y.SetZeroValue();

        size_t x_offset = 0, y_offset = 0;
        for(size_t n = 0; n < N; ++n) {
            ConvOperator(X, x_offset, _conv_shape, this->_Y, y_offset);
            x_offset += this->_in_neurons;
            y_offset += this->_out_neurons;
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, float learning_rate) override final {
        const auto& x_shape = X.GetShape();
        const auto& y_shape = dFdY.GetShape();
        const auto& N = x_shape.rows;
        const auto& out_neurons = this->_out_neurons;
        const auto& in_neurons = this->_in_neurons;
        const Shape& conv = _conv_shape;
        const Shape& out = _output_shape;
        if((y_shape.cols != out_neurons) || (y_shape.rows != N) || (in_neurons != x_shape.cols)) {
            throw "Conv backprop: wrong matrix shape";
        }

        this->_dFdX.Reshape(x_shape);
        this->_dFdX.SetZeroValue();
        this->_dFdW.SetZeroValue();

        for(size_t n = 0; n < N; ++n) {
            for(size_t k = 0; k < out.layers; ++k) {
                const auto w_offset = k * conv.Size();

                for(size_t r = 0; r < out.rows; ++r) {
                    for(size_t c = 0; c < out.cols; ++c) {
                        const auto ypos = n * out_neurons + k * out.LayerSize() + r * out.RowSize() + c;
                        const auto dfdy = dFdY[ypos];

                        const auto x_offset = n * in_neurons + r * _input_shape.RowSize() + c;

                        for(size_t layer = 0, w = 0; layer < conv.layers; ++layer) {
                            for(size_t j = 0; j < conv.rows; ++j) {
                                for(size_t i = 0; i < conv.cols; ++i, ++w) {
                                    const auto xpos = x_offset + layer * _input_shape.LayerSize() + j * _input_shape.RowSize() + i;
                                    this->_dFdX[xpos] += this->_W[w_offset + w] * dfdy;
                                    this->_dFdW[w_offset + w] += X[xpos] * dfdy;
                                }
                            }
                        }
                    }
                }
            }
        }

        const float delta = learning_rate / N;
        for(size_t k = 0; k < out.layers; ++k) {
            const auto w_offset = k * out.Size();
            for(size_t w = 0; w < conv.Size(); ++w) {
                this->_W[w_offset + w] -= delta * this->_dFdW[w_offset + w];
            }
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
        return CalcOutputNeurons(_input_shape, _conv_shape, _output_shape.layers);
    }

private:
    void ConvOperator(const Matrix<T>& X, size_t x_offset, const Shape& conv, Matrix<T>& Y, size_t y_offset) {
        const auto layer_size = _input_shape.LayerSize();
        const auto row_size = _input_shape.RowSize();

        for(size_t k = 0; k < _output_shape.layers; ++k) {
            const auto w_offset = k * conv.Size();

            for(size_t r = 0; r + conv.rows <= _input_shape.rows; ++r) {
                for(size_t c = 0; c + conv.cols <= _input_shape.cols; ++c) {
                    auto conv_offset = r * row_size + c;

                    float value = 0;
                    for(size_t layer = 0, w = 0; layer < conv.layers; ++layer) {
                        const auto offset = x_offset + layer * layer_size + conv_offset;
                        for(size_t j = 0; j < conv.rows; ++j) {
                            for(size_t i = 0; i < conv.cols; ++i, ++w) {
                                value += this->_W[w_offset + w] * X[offset + j*row_size + i];
                            }
                        }
                    }
                    Y[y_offset] = value;
                    y_offset++;
                }
            }
        }
    }

    static size_t CalcOutputNeurons(const Shape& input, const Shape& conv, const size_t& output_layers) {
        return output_layers * ((input.rows - conv.rows + 1) * (input.cols - conv.cols + 1));
    }

private:
    const Shape _input_shape;
    const Shape _output_shape;
    const Shape _conv_shape;
    Matrix3D<T> _W, _dFdW;
};

} //namespace layer
} //namespace nn
