#include "Common.h"
#include "layer/LinearLayer.h"

namespace nn {
namespace layer {

class LinearLayerTest: public ::testing::Test {
public:
    LinearLayerTest() = default;

protected:
    template <ExecutorType T>
    Matrix<T> TestForward(const Matrix<T>& X, size_t out_neurons) const {
        const auto& N = X.GetShape().rows;
        const auto& in_neurons = X.GetShape().cols;
        LinearLayer<T> layer(in_neurons, out_neurons);
        layer.Forward(X);
        layer.CopyDeviceToHost();

        const auto& Y = layer.GetOutput();
        const auto& W = layer.GetWeights();
        const auto& B = layer.GetBias();
        for(size_t n = 0; n < N; ++n) {
            for(size_t c = 0; c < out_neurons; ++c) {
                float value = B[c];
                for(size_t r = 0; r < in_neurons; ++r) {
                    value += X(n, r) * W(r, c);
                }
                AssertEqual(Y(n, c), value);
            }
        }
        return Y;
    }

    template <ExecutorType T>
    Matrix<T> TestBackprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float learning_rate) const {
        const auto shape = X.GetShape();
        const auto& N = shape.rows;
        const auto& in_neurons = shape.cols;
        const auto& out_neurons = dFdY.GetShape().cols;
        LinearLayer<T> layer(in_neurons, out_neurons);
        Matrix<T> W(Shape{in_neurons, out_neurons});
        W.CopyHostData(layer.GetWeights());
        layer.Forward(X);
        layer.CopyDeviceToHost();

        layer.Backprop(X, dFdY, learning_rate);
        layer.CopyDeviceToHost();

        const auto& dFdX = layer.GetGradient();
        for(size_t n = 0; n < N; ++n) {
            for(size_t c = 0; c < in_neurons; ++c) {
                float dfdx = 0;
                for(size_t k = 0; k < out_neurons; ++k) {
                    dfdx += dFdY(n, k) * W(c, k);
                }
                AssertEqual(dFdX(n, c), dfdx);
            }
        }
        return dFdX;
    }
};

TEST_F(LinearLayerTest, Forward) {
    for(auto begin = Clock::now(); Clock::now() - begin < 200ms; ) {
        const auto N = Rand<size_t>(1, 64);
        const auto in_neurons = Rand<size_t>(1, 256);
        const auto out_neurons = Rand<size_t>(1, 256);
        const Shape in_shape{N, in_neurons};

        Matrix<Cpu> X_cpu(in_shape);
        X_cpu.Xavier();

        Matrix<Cuda> X_cuda(in_shape);
        X_cuda.CopyHostData(X_cpu);
        X_cuda.CopyHostToDevice();

        const auto& Y_cpu = TestForward<Cpu>(X_cpu, out_neurons);
        const auto& Y_cuda = TestForward<Cuda>(X_cuda, out_neurons);

        ASSERT_EQ(Y_cpu.GetShape().rows, N);
        ASSERT_EQ(Y_cpu.GetShape().cols, out_neurons);
        ASSERT_EQ(Y_cuda.GetShape().rows, N);
        ASSERT_EQ(Y_cuda.GetShape().cols, out_neurons);
    }
}

TEST_F(LinearLayerTest, Backprop) {
    for(auto begin = Clock::now(); Clock::now() - begin < 200ms; ) {
        const auto N = Rand<size_t>(1, 64);
        const auto in_neurons = Rand<size_t>(1, 256);
        const auto out_neurons = Rand<size_t>(1, 256);
        const float learning_rate = Rand<float>(0.00001, 0.5);
        const Shape in_shape{N, in_neurons}, out_shape{N, out_neurons};

        Matrix<Cpu> X_cpu(in_shape), dFdY_cpu(out_shape);
        X_cpu.Xavier();
        dFdY_cpu.Xavier();

        Matrix<Cuda> X_cuda(in_shape), dFdY_cuda(out_shape);
        X_cuda.CopyHostData(X_cpu);
        dFdY_cuda.CopyHostData(dFdY_cpu);

        X_cuda.CopyHostToDevice();
        dFdY_cuda.CopyHostToDevice();

        const auto dFdX_cpu = TestBackprop<Cpu>(X_cpu, dFdY_cpu, learning_rate);
        const auto dFdX_cuda = TestBackprop<Cuda>(X_cuda, dFdY_cuda, learning_rate);

        ASSERT_EQ(dFdX_cpu.GetShape().rows, N);
        ASSERT_EQ(dFdX_cpu.GetShape().cols, in_neurons);
        ASSERT_EQ(dFdX_cuda.GetShape().rows, N);
        ASSERT_EQ(dFdX_cuda.GetShape().cols, in_neurons);
    }
}

} //namespace layer
} //namespace nn
