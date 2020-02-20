#pragma once

#include "Common.h"
#include "Layer.h"

namespace nn {
namespace layer {

//https://maziarraissi.github.io/teaching/3_backpropagation/

template <ExecutorType T = Cpu>
class LinearLayer : public Layer<T> {
public:
    explicit LinearLayer(const size_t input, const size_t output, const std::string& name = "")
        : Layer<T>(input, output, name)
        , _W(Shape{input, output})
        , _B(Shape{1, output})    {
        _W.XavierNormal();
        _W.CopyHostToDevice();
    }

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        if((this->_W.GetShape().rows != shape.cols)) {
            throw "LinearLayer forward: wrong matrix shape";
        }

        const auto N = shape.rows;
        this->_Y.Reshape(Shape{N, this->_out_neurons});
        this->_dFdX.Reshape(Shape{N, this->_in_neurons});

        // TODO: maybe move this functionality to Matrix
        const auto cols = this->_W.GetShape().cols;
        for(size_t r = 0, ix = 0, iy = 0; r < shape.rows; ++r, ix += shape.cols) {
            for(size_t c = 0; c < cols; ++c, ++iy) {
                float y = this->_B[c];
                for(size_t k = 0; k < shape.cols; ++k) {
                    y += X[ix + k] * this->_W(k, c);
                }
                this->_Y[iy] = y;
            }
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, float learning_rate) override final {
        const auto N = X.GetShape().rows;
        const auto& shape = this->_W.GetShape();
        const auto& in_neurons = this->_in_neurons;
        const auto& out_neurons = this->_out_neurons;
        if((dFdY.GetShape().cols != out_neurons) || (X.GetShape().rows != dFdY.GetShape().rows)) {
            throw "LinearLayer backprop: wrong matrix shape";
        }

        // dFdX[N, in_neurnons] = dFdY(N, out_neurons) * Transpose(W(in_neurons, out_neurons))
        // dFdW[in_neurnons, out_neurnons] = (1/N) * Transpose(X[N, in_neurnons]) * dFdY(N, out_neurons)
        // dFdB[1, out_neurnons] = (1/N) * 1[out_neurons, N] * dFdY(N, out_neurons)  -->  sum by columns

        // TODO: maybe move this functionality to Matrix
        for(size_t n = 0, ix = 0, iy = 0; n < N; ++n, iy += out_neurons) {
            for(size_t c = 0; c < in_neurons; ++c, ++ix) {
                float dfdx = 0;
                for(size_t k = 0; k < out_neurons; ++k) {
                    dfdx += dFdY[iy + k] * this->_W(c, k);
                }
                this->_dFdX[ix] = dfdx;
            }
        }

        const float delta = learning_rate / N;
        
        for(size_t r = 0; r < in_neurons; ++r) {
            for(size_t c = 0; c < out_neurons; ++c) {
                float dfdw = 0;
                size_t x_offset = r;
                size_t y_offset = c;
                for(size_t n = 0; n < N; ++n, x_offset += in_neurons, y_offset += out_neurons) {
                    dfdw += dFdY[y_offset] * X[x_offset];
                }
                this->_W(r, c) -= delta * dfdw;
            }
        }

        for(size_t c = 0; c < out_neurons; ++c) {
            float dfdb = 0;
            size_t y_offset = c;
            for(size_t n = 0; n < N; ++n, y_offset += out_neurons) {
                dfdb += dFdY[y_offset];
            }
            this->_B[c] -= delta * dfdb;
        }
    }

    void CopyDeviceToHost() const override final {
        Layer<T>::CopyDeviceToHost();
        _W.CopyDeviceToHost();
        _B.CopyDeviceToHost();
    }

    void Save(std::fstream& file) const override final {
        _W.CopyDeviceToHost();
        _B.CopyDeviceToHost();

        const auto shape = _W.GetShape();
        file.write(reinterpret_cast<char*>(_W.HostData()), shape.Size() * sizeof(float));
        file.write(reinterpret_cast<char*>(_B.HostData()), shape.cols * sizeof(float));
    }

    void Load(std::fstream& file) override final {
        const auto shape = _W.GetShape();
        file.read(reinterpret_cast<char*>(_W.HostData()), shape.Size() * sizeof(float));
        file.read(reinterpret_cast<char*>(_B.HostData()), shape.cols * sizeof(float));

        _W.CopyHostToDevice();
        _B.CopyHostToDevice();
    }

    const Matrix<T>& GetWeights() const /*override final*/ { return _W; }
    const Matrix<T>& GetBias() const /*override final*/ { return _B; }

private:
    Matrix<T> _W, _B;
};

} //namespace layer
} //namespace nn
