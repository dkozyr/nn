#pragma once

#include "layer/Layer.h"

namespace nn {
namespace activation {

namespace detail {

inline float Sigmoid(const float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

} // namespace detail

template <ExecutorType T = Cpu>
class Sigmoid : public layer::Layer<T> {
public:
	explicit Sigmoid(const size_t neurons, const std::string& name = "")
		: layer::Layer<T>(neurons, neurons, name)
	{}

	void Forward(const Matrix<T>& X) override final {
		const auto& shape = X.GetShape();
		if((this->_Y.GetShape().cols != shape.cols)) {
			throw "Sigmoid forward: wrong matrix shape";
		}

		this->_Y.Reshape(shape);
		this->_dFdX.Reshape(shape);

		const auto N = shape.Size();
		for(size_t i = 0; i < N; ++i) {
			this->_Y[i] = detail::Sigmoid(X[i]);
		}
	}

	void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float /*learning_rate*/) override final {
		const auto& shape = X.GetShape();
		if((shape.cols != dFdY.GetShape().cols) || (shape.cols != this->_Y.GetShape().cols)) {
			throw "Sigmoid backprop: wrong matrix shape";
		}

		const auto N = shape.Size();
		for(size_t i = 0; i < N; ++i) {
			const auto& sigmoid = this->_Y[i];
			this->_dFdX[i] = dFdY[i] * sigmoid * (1.0f - sigmoid);
		}
	}
};

} //namespace layer
} //namespace nn
