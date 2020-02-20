#include "Common.h"
#include "activation/ReLu.h"

namespace nn {
namespace activation {

class ReLuActivationTest: public ::testing::Test {
public:
    ReLuActivationTest() = default;

protected:
    template <ExecutorType T>
    Matrix<T> TestForward(const Matrix<T>& X) const {
        const auto& shape = X.GetShape();
        ReLu<T> layer(shape.cols);
        layer.Forward(X);
        layer.CopyDeviceToHost();

        const auto& Y = layer.GetOutput();
        for(size_t i = 0; i < shape.Size(); ++i) {
            AssertEqual(Y[i], std::max(0.0f, X[i]));
        }
        return Y;
    }

    template <ExecutorType T>
    Matrix<T> TestBackprop(const Matrix<T>& X, const Matrix<T>& dFdY) const {
        const auto& shape = X.GetShape();
        ReLu<T> layer(shape.cols);
        layer.Forward(X);
        layer.Backprop(X, dFdY, 0.0);
        layer.CopyDeviceToHost();

        const auto& dFdX = layer.GetGradient();
        for(size_t i = 0; i < shape.Size(); ++i) {
            AssertEqual(dFdX[i], dFdY[i] * (X[i] > 0 ? 1.0 : 0));
        }
        return dFdX;
    }
};

TEST_F(ReLuActivationTest, Forward) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand<size_t>(1, kMaxBatch);
        const auto cols = Rand<size_t>(1, 256);
        const Shape shape{rows, cols};

        Matrix<Cpu> X_cpu(shape);
        X_cpu.Xavier();

        Matrix<Cuda> X_cuda(shape);
        X_cuda.CopyHostData(X_cpu);
        X_cuda.CopyHostToDevice();

        const auto Y_cpu = TestForward<Cpu>(X_cpu);
        const auto Y_cuda = TestForward<Cuda>(X_cuda);

        ASSERT_EQ(Y_cpu.GetShape().rows, rows);
        ASSERT_EQ(Y_cpu.GetShape().cols, cols);
        ASSERT_EQ(Y_cuda.GetShape().rows, rows);
        ASSERT_EQ(Y_cuda.GetShape().cols, cols);
        for(size_t i = 0; i < rows * cols; ++i) {
            AssertEqual(Y_cpu[i], Y_cuda[i]);
        }
    }
}

TEST_F(ReLuActivationTest, Backprop) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand<size_t>(1, kMaxBatch);
        const auto cols = Rand<size_t>(1, 256);
        const Shape shape{rows, cols};

        Matrix<Cpu> A_cpu(shape), dFdY_cpu(shape);
        A_cpu.Xavier();
        dFdY_cpu.Xavier();

        Matrix<Cuda> A_cuda(shape), dFdY_cuda(shape);
        A_cuda.CopyHostData(A_cpu);
        dFdY_cuda.CopyHostData(dFdY_cpu);

        A_cuda.CopyHostToDevice();
        dFdY_cuda.CopyHostToDevice();

        const auto dFdX_cpu = TestBackprop<Cpu>(A_cpu, dFdY_cpu);
        const auto dFdX_cuda = TestBackprop<Cuda>(A_cuda, dFdY_cuda);

        ASSERT_EQ(dFdX_cpu.GetShape().rows, rows);
        ASSERT_EQ(dFdX_cpu.GetShape().cols, cols);
        ASSERT_EQ(dFdX_cuda.GetShape().rows, rows);
        ASSERT_EQ(dFdX_cuda.GetShape().cols, cols);
        for(size_t i = 0; i < rows * cols; ++i) {
            AssertEqual(dFdX_cpu[i], dFdX_cuda[i]);
        }
    }
}

} //namespace activation
} //namespace nn
