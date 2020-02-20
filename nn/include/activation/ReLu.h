#pragma once

#include "layer/Layer.h"

namespace nn {
namespace activation {

template <ExecutorType T = Cpu>
class ReLu : public layer::Layer<T> {
public:
	explicit ReLu(const size_t neurons, const std::string& name = "")
		: layer::Layer<T>(neurons, neurons, name)
	{}

	void Forward(const Matrix<T>& X) override final {
		const auto& shape = X.GetShape();
		if((this->_Y.GetShape().cols != shape.cols)) {
			throw "ReLu forward: wrong matrix shape";
		}

		const auto N = shape.rows;
		this->_Y.Reshape(shape);
		this->_dFdX.Reshape(shape);

		const auto size = shape.Size();
		for(size_t i = 0; i < size; ++i) {
			this->_Y[i] = std::max(0.0f, X[i]);
		}
	}

	void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float /*learning_rate*/) override final {
		const auto& shape = X.GetShape();
		if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
		   (shape.cols != this->_Y.GetShape().cols) || (shape.rows != this->_Y.GetShape().rows)) {
			throw "ReLu backprop: wrong matrix shape";
		}

		const auto size = shape.Size();
		for(size_t i = 0; i < size; ++i) {
			if(X[i] > 0) {
				this->_dFdX[i] = dFdY[i];
			} else {
				this->_dFdX[i] = 0.0f;
			}
		}
	}
};

} //namespace activation
} //namespace nn
