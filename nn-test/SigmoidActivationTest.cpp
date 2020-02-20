#include "Common.h"
#include "activation/Sigmoid.h"

namespace nn {
namespace activation {

class SigmoidActivationTest: public ::testing::Test {
public:
    SigmoidActivationTest() = default;

protected:
    template <ExecutorType T>
    Matrix<T> TestForward(const Matrix<T>& X) const {
        const auto& shape = X.GetShape();
        Sigmoid<T> layer(shape.cols);
        layer.Forward(X);
        layer.CopyDeviceToHost();

        const auto& Y = layer.GetOutput();
        for(size_t i = 0; i < shape.Size(); ++i) {
            AssertEqual(Y[i], detail::Sigmoid(X[i]));
        }
        return Y;
    }

    template <ExecutorType T>
    Matrix<T> TestBackprop(const Matrix<T>& X, const Matrix<T>& dFdY) const {
        const auto& shape = X.GetShape();
        Sigmoid<T> layer(shape.cols);
        layer.Forward(X);
        layer.Backprop(X, dFdY, 0.0);
        layer.CopyDeviceToHost();

        const auto& dFdX = layer.GetGradient();
        for(size_t i = 0; i < shape.Size(); ++i) {
            const auto sigmoid = detail::Sigmoid(X[i]);
            AssertEqual(dFdX[i], dFdY[i] * sigmoid * (1.0 - sigmoid));
        }
        return dFdX;
    }
};

TEST_F(SigmoidActivationTest, Forward) {
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

TEST_F(SigmoidActivationTest, Backprop) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand<size_t>(1, kMaxBatch);
        const auto cols = Rand<size_t>(1, 256);
        const Shape shape{rows, cols};

        Matrix<Cpu> X_cpu(shape), dFdY_cpu(shape);
        X_cpu.Xavier();
        dFdY_cpu.Xavier();

        Matrix<Cuda> X_cuda(shape), dFdY_cuda(shape);
        X_cuda.CopyHostData(X_cpu);
        dFdY_cuda.CopyHostData(dFdY_cpu);

        X_cuda.CopyHostToDevice();
        dFdY_cuda.CopyHostToDevice();

        const auto dFdX_cpu = TestBackprop<Cpu>(X_cpu, dFdY_cpu);
        const auto dFdX_cuda = TestBackprop<Cuda>(X_cuda, dFdY_cuda);

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
