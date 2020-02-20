#pragma once

#include "layer/Layer.h"

namespace nn {
namespace activation {

template <ExecutorType T = Cpu>
class Softmax : public layer::Layer<T> {
public:
	explicit Softmax(const size_t neurons, const std::string& name = "")
		: layer::Layer<T>(neurons, neurons, name)
	{}

	void Forward(const Matrix<T>& X) override final {
		const auto& shape = X.GetShape();
		if((this->_Y.GetShape().cols != shape.cols)) {
			throw "Softmax forward: wrong matrix shape";
		}

		const auto& N = shape.rows;
		const auto& out_neurons = this->_out_neurons;

		this->_Y.Reshape(Shape{N, out_neurons});
		this->_dFdX.Reshape(shape);

        auto ptr = X.HostData();
		for(size_t n = 0, offset = 0; n < N; ++n, offset += out_neurons) {
            const auto shift = *std::max_element(ptr, ptr + offset);

            float sum = 0;
            for(size_t i = 0; i < out_neurons; ++i) {
                this->_Y[offset + i] = std::exp(X[offset + i] - shift);
                sum += this->_Y[offset + i];
            }

            const float factor = 1.0 / sum;
            for(size_t i = 0; i < out_neurons; ++i) {
                this->_Y[offset + i] *= factor;
            }
        }
	}

	void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float /*learning_rate*/) override final {
		const auto& shape = X.GetShape();
		if((shape.cols != dFdY.GetShape().cols) || (shape.cols != this->_Y.GetShape().cols)) {
			throw "Softmax backprop: wrong matrix shape";
		}

		const auto N = shape.Size();
		memcpy(this->_dFdX.HostData(), dFdY.HostData(), N * sizeof(float));
	}
};

} //namespace activation
} //namespace nn
