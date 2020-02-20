#pragma once

#include "Cost.h"

namespace nn {
namespace cost {

template <ExecutorType T = Cpu>
class CrossEntropy : public Cost<T> {
public:
	explicit CrossEntropy(const size_t neurons, const std::string& name = "")
		: Cost<T>(neurons, name) {}

	void Evaluate(const Matrix<T>& Yh, const Matrix<T>& Y) override final {
        const auto& shape = Yh.GetShape();
        if((shape.rows != Y.GetShape().rows) || (Y.GetShape().cols != shape.cols)) {
            throw "CrossEntropy Evaluate: wrong shape size";
        }
        this->_dFdX.Reshape(shape);

        const auto N = shape.Size();
        for(size_t i = 0; i < N; ++i) {
            if(Y[i] > 0) {
                if(Yh[i] > 0) {
                    this->_loss.value -= Y[i] * std::log(Yh[i]); // Y[i] == 1 (?)
                } else {
                    this->_loss.value -= 1e3;
                }
            }
            this->_dFdX[i] = Yh[i] - Y[i];
        }
        this->_loss.samples += shape.rows;
    }

	float GetLoss() const override final {
        return this->_loss.value;
    }
};

} //namespace cost
} //namespace nn
