#include "Common.h"
#include "layer/Conv3D.h"

namespace nn {
namespace layer {

class Conv3DTest: public ::testing::Test {
public:
    Conv3DTest() = default;

protected:
    template <ExecutorType T>
    Matrix<T> TestForward(const Matrix<T>& X, const Shape& input, size_t conv_size) const {
        const auto& N = X.GetShape().rows;
        Conv3D<T> layer(input, Shape{conv_size, conv_size});
        layer.Forward(X);
        layer.CopyDeviceToHost();

        const auto Y = layer.GetOutput();
        const auto out_neurons = layer.GetOutputNeurons();
        for(size_t n = 0; n < N; ++n) {
            for(size_t i = 0; i < out_neurons; ++i) {
                AssertEqual(Y(n, i), X[0]);
            }
        }
        return Y;
    }
};

TEST_F(Conv3DTest, DISBLED_Forward) {
    const auto N = 16;
    const auto conv_size = 2;
    const auto rgb_layers = 3;
    const Shape input{rgb_layers, 10, 10};

    Matrix<Cpu> X_cpu(Shape{N, input.Size()}, Rand(-3.0f, 3.0f));
    Matrix<Cuda> X_cuda(Shape{N, input.Size()});
    X_cuda.CopyHostData(X_cpu);
    X_cuda.CopyHostToDevice();

    const auto& Y_cpu = TestForward<Cpu>(X_cpu, input, conv_size);
    const auto& Y_cuda = TestForward<Cuda>(X_cuda, input, conv_size);

    const auto out_neurons = /*input.layers **/ (input.rows - conv_size + 1) * (input.cols - conv_size + 1);
    ASSERT_EQ(Y_cpu.GetShape().rows, N);
    ASSERT_EQ(Y_cpu.GetShape().cols, out_neurons);
    ASSERT_EQ(Y_cuda.GetShape().rows, N);
    ASSERT_EQ(Y_cuda.GetShape().cols, out_neurons);
}

TEST_F(Conv3DTest, DISABLED_Forward) {
    for(auto begin = Clock::now(); Clock::now() - begin < 200ms; ) {
        const auto N = Rand<size_t>(1, 64);
        const auto conv_size = Rand<size_t>(2, 5);
        const auto rgb_layers = 3;
        const Shape input{rgb_layers, Rand<size_t>(conv_size, 64), Rand<size_t>(conv_size, 64)};

        Matrix<Cpu> X_cpu(Shape{N, input.Size()}, Rand(-3.0f, 3.0f));
        Matrix<Cuda> X_cuda(Shape{N, input.Size()});
        X_cuda.CopyHostData(X_cpu);
        X_cuda.CopyHostToDevice();

        const auto& Y_cpu = TestForward<Cpu>(X_cpu, input, conv_size);
        const auto& Y_cuda = TestForward<Cuda>(X_cuda, input, conv_size);

        const auto out_neurons = /*input.layers **/ (input.rows - conv_size + 1) * (input.cols - conv_size + 1);
        ASSERT_EQ(Y_cpu.GetShape().rows, N);
        ASSERT_EQ(Y_cpu.GetShape().cols, out_neurons);
        ASSERT_EQ(Y_cuda.GetShape().rows, N);
        ASSERT_EQ(Y_cuda.GetShape().cols, out_neurons);
    }
}

} //namespace layer
} //namespace nn
