#pragma once

#include "Common.h"
#include "Layer.h"

namespace nn {
namespace layer {

// https://costapt.github.io/2016/07/09/batch-norm-alt/

template <ExecutorType T = Cpu>
class BatchNorm : public Layer<T> {
    static constexpr float eps = 0.01f;

public:
    explicit BatchNorm(const size_t neurons, const std::string& name = "")
        : Layer<T>(neurons, neurons, name)
        , _xhat(Shape{kMaxBatch, neurons})
        , _mean(Shape{1, neurons})
        , _sigma(Shape{1, neurons})
        , _gamma(Shape{1, neurons}, 1.0f)
        , _beta(Shape{1, neurons})    {
    }

    void Forward(const Matrix<T>& X) override final {
        const auto& shape = X.GetShape();
        const auto& N = shape.rows;
        const auto& neurons = this->_Y.GetShape().cols;
        if((neurons != shape.cols) || (N < 4)) {
            throw "BatchNorm forward: wrong matrix shape";
        }

        this->_Y.Reshape(shape);
        this->_dFdX.Reshape(shape);
        this->_xhat.Reshape(shape);
        
        CalculateMean(X, N, neurons);
        CalculateStdDev(X, N, neurons);

        for(size_t r = 0; r < N; ++r) {
            for(size_t c = 0; c < neurons; ++c) {
                if(this->_sigma[c] > eps) {
                    this->_xhat(r, c) = (X(r, c) - this->_mean[c]) / this->_sigma[c];
                    this->_Y(r, c) = this->_gamma[c] * this->_xhat(r, c) + this->_beta[c];
                } else {
                    this->_xhat(r, c) = 0;
                    this->_Y(r, c) = this->_beta[c];
                }
            }
        }
    }

    void Backprop(const Matrix<T>& X, const Matrix<T>& dFdY, float learning_rate) override final {
        const auto& shape = X.GetShape();
        const auto& N = shape.rows;
        const auto& neurons = this->_Y.GetShape().cols;
        if((shape.cols != dFdY.GetShape().cols) || (shape.rows != dFdY.GetShape().rows) ||
           (shape.cols != this->_Y.GetShape().cols) || (shape.rows > this->_Y.GetShape().rows)) {
            throw "BatchNorm backprop: wrong matrix shape";
        }

        // def batchnorm_backward_alt(dout, cache):
        //     gamma, xhat, istd = cache
        //     N, _ = dout.shape
        //     dbeta = np.sum(dout, axis=0)
        //     dgamma = np.sum(xhat * dout, axis=0)
        //     dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)
        //     return dx, dgamma, dbeta

        for(size_t c = 0; c < neurons; ++c) {
            float dFdGamma = 0.0f, dFdBeta = 0.0f;
            for(size_t r = 0; r < N; ++r) {
                dFdGamma += dFdY(r, c) * this->_xhat(r, c);
                dFdBeta += dFdY(r, c);
            }

            const float factor = this->_gamma[c] * this->_sigma[c] / N;
            for(size_t r = 0; r < N; ++r) {
                this->_dFdX(r, c) = factor * (N * dFdY(r, c) - dFdGamma * this->_xhat(r, c) - dFdBeta);
            }
    
            const float delta = learning_rate / N;
            this->_gamma[c] -= delta * dFdGamma;
            this->_beta[c] -= delta * dFdBeta;
        }
    }

    void Save(std::fstream& file) const override final {
        _gamma.Save(file);
        _beta.Save(file);
    }

    void Load(std::fstream& file) override final {
        _gamma.Load(file);
        _beta.Load(file);
    }

private:
    void CalculateMean(const Matrix<T>& X, const size_t N, const size_t neurons) {
        const float factor = 1.0f / N;
        for(size_t c = 0; c < neurons; ++c) {
            float mean = 0.0f;
            for(size_t r = 0; r < N; ++r) {
                mean += X(r, c);
            }
            this->_mean[c] = factor * mean;
        }
    }

    void CalculateStdDev(const Matrix<T>& X, const size_t N, const size_t neurons) {
        const float factor = 1.0f / N;
        for(size_t c = 0; c < neurons; ++c) {
            float sigma2 = 0.0f;
            for(size_t r = 0; r < N; ++r) {
                const auto x_zero_mean = X(r, c) - this->_mean[c];
                sigma2 += x_zero_mean * x_zero_mean;
            }
            this->_sigma[c] = sqrt(factor * sigma2 + eps);
        }
    }

private:
    Matrix<T> _xhat;
    Matrix<T> _mean, _sigma;
    Matrix<T> _gamma, _beta;
};

} //namespace layer
} //namespace nn
