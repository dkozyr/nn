#pragma once

#include "Matrix.h"

namespace nn {
namespace layer {

template <ExecutorType T>
class Layer {
public:
    Layer(const size_t inputs, const size_t outputs, const std::string& name)
        : _name(name)
		, _in_neurons(inputs)
		, _out_neurons(outputs)
        , _Y(Shape{kMaxBatch, outputs})
        , _dFdX(Shape{kMaxBatch, inputs})
    {}
	virtual ~Layer() = default;

	const std::string& GetName() const { return _name; };
    const Matrix<T>& GetOutput() const { return _Y; }
    const Matrix<T>& GetGradient() const { return _dFdX; }

	size_t GetInNeurons() const { return _in_neurons; }
	size_t GetOutNeurons() const { return _out_neurons; }

	virtual void Predict(const Matrix<T>& X) { Forward(X); }
	virtual void Forward(const Matrix<T>& X) = 0;
	virtual void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float learning_rate) = 0;

	virtual void CopyDeviceToHost() const {
		_Y.CopyDeviceToHost();
		_dFdX.CopyDeviceToHost();
	}

	virtual void Save(std::fstream& file) const {}
	virtual void Load(std::fstream& file) {}

protected:
	const std::string _name;
	const size_t _in_neurons;
	const size_t _out_neurons;
	Matrix<T> _Y, _dFdX;
};

} //namespace layer
} //namespace nn
