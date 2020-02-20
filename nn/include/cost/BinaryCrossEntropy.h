#pragma once

#include "Cost.h"

namespace nn {
namespace cost {

template <ExecutorType T = Cpu>
class BinaryCrossEntropy : public Cost<T> {
public:
	explicit BinaryCrossEntropy(const std::string& name = "")
		: Cost<T>(1, name) {}

	void Evaluate(const Matrix<T>& Yh, const Matrix<T>& Y) override final {
        const auto& shape = Yh.GetShape();
        if((shape.rows != Y.GetShape().rows) || (Y.GetShape().cols != 1) || (shape.cols != 1)) {
            throw "BinaryCrossEntropy Evaluate: wrong shape size";
        }
        this->_dFdX.Reshape(shape);

        const auto N = shape.rows;
        for(size_t i = 0; i < N; ++i) {
            const auto& y = Y[i];
            const auto& yh = Yh[i];
            if(yh == y) {
                this->_dFdX[i] = 0;
            } else if(yh == 0) {
                this->_loss.value += 10.f;
                this->_dFdX[i] = +10.0f;
            } else if(yh == 1.0f) {
                this->_loss.value += 10.f;
                this->_dFdX[i] = -10.0f;
            } else {
                this->_loss.value -= y * std::log(yh) + (1.0f - y) * std::log(1.0f - yh);
                this->_dFdX[i] = -(y / yh - (1.0f - y) / (1.0f - yh));
            }
        }
        this->_loss.samples += N;
    }

	float GetLoss() const override final {
        const auto& loss = this->_loss;
        return (loss.samples != 0) ? loss.value / loss.samples : 0;
    }
};

} //namespace cost
} //namespace nn
