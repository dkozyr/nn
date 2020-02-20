#pragma once

#include "Matrix.h"

namespace nn {
namespace cost {

template <ExecutorType T>
class Cost {
public:
    Cost(const size_t neurons, const std::string& name)
        : _name(name)
        , _neurons(neurons)
        , _dFdX(Shape{kMaxBatch, neurons})
    {}
    virtual ~Cost() = default;

    const std::string& GetName() const { return _name; };
    const Matrix<T>& GetGradient() const { return _dFdX; }

    virtual void Evaluate(const Matrix<T>& Yh, const Matrix<T>& Y) = 0;
    virtual float GetLoss() const = 0;
    void ResetLoss() { _loss = Loss{0, 0}; }

    void CopyDeviceToHost() const { _dFdX.CopyDeviceToHost();}

protected:
    struct Loss {
        float value = 0;
        size_t samples = 0;
    };

protected:
    const std::string _name;
    const size_t _neurons;
    Matrix<T> _dFdX;
    Loss _loss;
};

} //namespace cost
} //namespace nn
