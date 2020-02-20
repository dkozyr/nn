#pragma once

#include "layer/Layer.h"

namespace nn {
namespace activation {

namespace detail {

inline float Tanh(const float x) {
    const float exp2x = std::exp(x + x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

} // namespace detail

template <ExecutorType T = Cpu>
class Tanh : public layer::Layer<T> {
public:
    explicit Tanh(const size_t neurons, const std::string& name = "")
        : layer::Layer<T>(neurons, neurons, name)
    {}

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        if((this->_Y.GetShape().cols != shape.cols)) {
            throw "Tanh forward: wrong matrix shape";
        }

        this->_Y.Reshape(shape);
        this->_dFdX.Reshape(shape);

        const auto N = shape.Size();
        for(size_t i = 0; i < N; ++i) {
            this->_Y[i] = detail::Tanh(X[i]);
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float /*learning_rate*/) override final {
        const auto& shape = X.GetShape();
        if((shape.cols != dFdY.GetShape().cols) || (shape.cols != this->_Y.GetShape().cols)) {
            throw "Tanh backprop: wrong matrix shape";
        }

        const auto N = shape.Size();
        for(size_t i = 0; i < N; ++i) {
            const auto& tanh = this->_Y[i];
            this->_dFdX[i] = dFdY[i] * (1.0f - tanh * tanh);
        }
    }
};

} //namespace activation
} //namespace nn
