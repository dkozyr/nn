#pragma once

#include "layer/LinearLayer.h"
#include "layer/Conv.h"
#include "layer/Conv3D.h"
#include "layer/Dropout.h"
#include "layer/BatchNorm.h"
#include "layer/MaxPool.h"
#include "activation/ReLu.h"
#include "activation/Sigmoid.h"
#include "activation/Tanh.h"
#include "activation/Softmax.h"
#include "cost/BinaryCrossEntropy.h"
#include "cost/CrossEntropy.h"

namespace nn {

template <ExecutorType T = Cpu>
class Network {
public:
    Network() : _name("") {}

    explicit Network(const std::string& name) : _name(name) {
    }

    void AddLinearLayer(const size_t input, const size_t output, const std::string& name = "") {
        _layers.emplace_back(new layer::LinearLayer<T>(input, output, name));
    }

    void AddReLu(const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new activation::ReLu<T>(neurons, name));
    }

    void AddSigmoid(const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new activation::Sigmoid<T>(neurons, name));
    }

    void AddTanh(const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new activation::Tanh<T>(neurons, name));
    }

    void AddSoftmax(const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new activation::Softmax<T>(neurons, name));
    }

    void AddDropout(const float probability, const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new layer::Dropout<T>(neurons, probability, name));
    }

    void AddBatchNorm(const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new layer::BatchNorm<T>(neurons, name));
    }

    void AddMaxPool(const Shape& input, const Shape& window, const size_t stride = 1, const std::string& name = "") {
        const auto neurons = _layers.back()->GetOutNeurons();
        _layers.emplace_back(new layer::MaxPool<T>(input, window, stride, name));
    }

    void AddConv(const Shape& input, const Shape& conv, const std::string& name = "") {
        _layers.emplace_back(new layer::Conv<T>(input, conv, name));
    }

    void AddConv3D(const Shape& input, const Shape& conv, const std::string& name = "") {
        _layers.emplace_back(new layer::Conv3D<T>(input, conv, name));
    }

    void AddBinaryCrossEntropy(const std::string& name = "") {
        _cost = std::shared_ptr<cost::Cost<T>>(new cost::BinaryCrossEntropy<T>(name));
    }

    void AddCrossEntropy(const size_t output, const std::string& name = "") {
        _cost = std::shared_ptr<cost::Cost<T>>(new cost::CrossEntropy<T>(output, name));
    }

    const Matrix<T>& Predict(const Matrix<T>& X) const {
        const auto num_layers = _layers.size();
        for(size_t i = 0; i < num_layers; ++i) {
            const auto& input = (i == 0) ? X : _layers[i-1]->GetOutput();
            _layers[i]->Predict(input);
        }
        _layers.back()->CopyDeviceToHost();
        return _layers.back()->GetOutput();
    }

    float Accuracy(const Matrix<T>& X, const Matrix<T>& Y) {
        const auto N = X.GetShape().rows;
        const auto neurons = Y.GetShape().cols;
        const size_t chunk_size = 16;

        float correct = 0;
        for(size_t i = 0; i < N; i += chunk_size) {
            const auto batch_size = (i + chunk_size > N) ? (N - i) : chunk_size;
            const auto& X_batch = X.GetSubMatrix(i, batch_size);
            const auto& Y_batch = Y.GetSubMatrix(i, batch_size);
            const auto& Yh = Predict(X_batch);
            Yh.CopyDeviceToHost();

            for(size_t n = 0; n < batch_size; ++n) {
                if(neurons == 1) {
                    if(Y_batch[n] == std::roundf(Yh[n])) {
                        ++correct;
                    }
                } else {
                    size_t idx = 0;
                    for(size_t i = 1; i < neurons; ++i) {
                        if(Yh(n, idx) < Yh(n, i)) {
                            idx = i;
                        }
                    }
                    if(Y_batch(n, idx) > 0) {
                        ++correct;
                    }
                }
            }
        }
        return correct / N;
    }

    float Train(const Matrix<T>& X, const Matrix<T>& Y, const float learning_rate, const size_t train_batch_size = 32) {
        const auto num_samples = X.GetShape().rows;
        const auto num_layers = _layers.size();
        _cost->ResetLoss();
        for(size_t j = 0; j < num_samples; j += train_batch_size) {
            const auto batch_size = (j + train_batch_size > num_samples) ? (num_samples - j) : train_batch_size;
            const auto X_batch = X.GetSubMatrix(j, batch_size);
            const auto Y_batch = Y.GetSubMatrix(j, batch_size);

            Forward(X_batch);

            const auto N = X_batch.GetShape().rows;
            const auto& Yh = _layers.back()->GetOutput();

            _cost->Evaluate(Yh, Y_batch);

            for(int i = num_layers - 1; i >= 0; --i) {
                const auto input = (i == 0) ? X_batch : _layers[i-1]->GetOutput();
                const auto gradient = (i == num_layers - 1) ? _cost->GetGradient() : _layers[i + 1]->GetGradient();
                _layers[i]->Backprop(input, gradient, learning_rate);
            }
        }
        return _cost->GetLoss();
    }

    void Save(const std::string& path) const {
        auto file = std::fstream(path, std::ios::out | std::ios::binary);
        for(auto& layer: _layers) {
            layer->Save(file);
        }
        file.close();
    }

    void Load(const std::string& path) {
        auto file = std::fstream(path, std::ios::in | std::ios::binary);
        for(auto& layer: _layers) {
            layer->Load(file);
        }
        file.close();
    }

private:
    void Forward(const Matrix<T>& X) {
        const auto num_layers = _layers.size();
        for(size_t i = 0; i < num_layers; ++i) {
            const auto& input = (i == 0) ? X : _layers[i-1]->GetOutput();
            _layers[i]->Forward(input);
        }
    }

private:
    const std::string _name;
    std::vector<std::shared_ptr<layer::Layer<T>>> _layers;
    std::shared_ptr<cost::Cost<T>> _cost;
};

} //namespace nn
