#include "Common.h"
#include "layer/Dropout.h"

namespace nn {
namespace layer {

class DropoutTest: public ::testing::Test {
public:
    DropoutTest() = default;

protected:
    template <ExecutorType T>
    Matrix<T> TestForward(const Matrix<T>& X, const float probability) const {
        const auto& neurons = X.GetShape().cols;
        Dropout<T> layer(neurons, probability);
        layer.Forward(X);
        layer.CopyDeviceToHost();

        const auto& Y = layer.GetOutput();
        const auto N = X.GetShape().Size();
        float dropped_count = 0;
        for(size_t i = 0; i < N; ++i) {
            if((Y[i] == 0) && (std::abs(X[i] - Y[i]) > 0.00001)) {
                dropped_count++;
            }
        }
        const float estimation = dropped_count / N;
        EXPECT_NEAR(estimation, probability, 0.02f);
        return Y;
    }

    template <ExecutorType T>
    Matrix<T> TestBackprop(const Matrix<T>& X, const Matrix<T>& dFdY, const float probability) const {
        const auto& neurons = X.GetShape().cols;
        Dropout<T> layer(neurons, probability);
        layer.Forward(X);
        layer.Backprop(X, dFdY, 0);
        layer.CopyDeviceToHost();

        const auto& Y = layer.GetOutput();
        const auto& dFdX = layer.GetGradient();
        const auto N = X.GetShape().Size();
        float dropped_count = 0;
        for(size_t i = 0; i < N; ++i) {
            if((Y[i] == 0) && (std::abs(X[i] - Y[i]) > 0.00001) && (std::abs(dFdX[i] - dFdY[i]) > 0.00001)) {
                dropped_count++;
            }
        }
        const float estimation = dropped_count / N;
        EXPECT_NEAR(estimation, probability, 0.02f);
        return dFdX;
    }
};

TEST_F(DropoutTest, Forward) {
    for(auto begin = Clock::now(); Clock::now() - begin < 200ms; ) {
        const auto N = 8;
        const auto neurons = 1024;
        const Shape shape{N, neurons};
        const auto probability = Rand<float>(0.01, 0.5);

        Matrix<Cpu> X_cpu(shape);
        X_cpu.Xavier();

        Matrix<Cuda> X_cuda(shape);
        X_cuda.CopyHostData(X_cpu);
        X_cuda.CopyHostToDevice();

        const auto& Y_cpu = TestForward<Cpu>(X_cpu, probability);
        const auto& Y_cuda = TestForward<Cuda>(X_cuda, probability);

        ASSERT_EQ(Y_cpu.GetShape().rows, N);
        ASSERT_EQ(Y_cpu.GetShape().cols, neurons);
        ASSERT_EQ(Y_cuda.GetShape().rows, N);
        ASSERT_EQ(Y_cuda.GetShape().cols, neurons);
    }
}

TEST_F(DropoutTest, Backprop) {
    for(auto begin = Clock::now(); Clock::now() - begin < 200ms; ) {
        const auto N = 8;
        const auto neurons = 1024;
        const Shape shape{N, neurons};
        const auto probability = Rand<float>(0.01, 0.5);

        Matrix<Cpu> X_cpu(shape), dFdY_cpu(shape);
        X_cpu.Xavier();
        dFdY_cpu.Xavier();

        Matrix<Cuda> X_cuda(shape), dFdY_cuda(shape);
        X_cuda.CopyHostData(X_cpu);
        dFdY_cuda.CopyHostData(dFdY_cpu);

        X_cuda.CopyHostToDevice();
        dFdY_cuda.CopyHostToDevice();

        const auto dFdX_cpu = TestBackprop<Cpu>(X_cpu, dFdY_cpu, probability);
        const auto dFdX_cuda = TestBackprop<Cuda>(X_cuda, dFdY_cuda, probability);

        ASSERT_EQ(dFdX_cpu.GetShape().rows, N);
        ASSERT_EQ(dFdX_cpu.GetShape().cols, neurons);
        ASSERT_EQ(dFdX_cuda.GetShape().rows, N);
        ASSERT_EQ(dFdX_cuda.GetShape().cols, neurons);
    }
}

} //namespace layer
} //namespace nn
