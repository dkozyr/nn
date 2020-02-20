#pragma once

#include "Common.h"
#include "Layer.h"

namespace nn {
namespace layer {

template <ExecutorType T = Cpu>
class Dropout : public Layer<T> {
public:
	explicit Dropout(const size_t neurons, const float probability, const std::string& name = "")
		: Layer<T>(neurons, neurons, name)
		, _probability(probability)
	{}

	void Predict(const Matrix<T>& X) override final {
		const auto& shape = X.GetShape();
		if((this->_Y.GetShape().cols != shape.cols)) {
			throw "Dropout forward: wrong matrix shape";
		}

		this->_Y.Reshape(shape);
		this->_dFdX.Reshape(shape);

		const auto N = shape.Size();
		memcpy(this->_Y.HostData(), X.HostData(), N * sizeof(float));
	}

	void Forward(const Matrix<T>& X) override final {
		const auto& shape = X.GetShape();
		if((this->_Y.GetShape().cols != shape.cols)) {
			throw "Dropout forward: wrong matrix shape";
		}

		this->_Y.Reshape(shape);
		this->_dFdX.Reshape(shape);

		const auto N = shape.Size();
		for(size_t i = 0; i < N; ++i) {
			if(Rand(0.0f, 1.0f) < _probability) {
				this->_Y[i] = 0;
				this->_dFdX[i] = -1.0f;
			} else {
				this->_Y[i] = X[i];
				this->_dFdX[i] = 1.0f;
			}
		}
	}

	void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, float /*learning_rate*/) override final {
		const auto& shape = X.GetShape();
		if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
		   (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
			throw "Dropout backprop: wrong matrix shape";
		}

		const auto N = shape.Size();
		for(size_t i = 0; i < N; ++i) {
			if(this->_dFdX[i] > 0) {
				this->_dFdX[i] = dFdY[i];
			} else {
				this->_dFdX[i] = 0.0f;
			}
		}
	}

private:
	const float _probability;
};

} //namespace layer
} //namespace nn
