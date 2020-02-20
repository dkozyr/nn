#include "Common.h"
#include "cost/CrossEntropy.h"

namespace nn {
namespace cost {

constexpr float kEps = 1e-3;

class CrossEntropyTest: public ::testing::Test {
public:
    CrossEntropyTest() = default;

protected:
    template <ExecutorType T>
    static float TestCrossEntropy(const size_t rows, const size_t cols, const float prob) {
        Matrix<T> Yh(Shape{rows, cols}, prob), Y(Shape{rows, cols}, 1.0f);
        CrossEntropy<T> cost(cols);
        cost.Evaluate(Yh, Y);
        const auto estimation = -std::log(prob) * rows * cols;
        AssertEqual(estimation, cost.GetLoss());
        return cost.GetLoss();
    }

    template <ExecutorType T>
    static Matrix<T> TestCrossEntropyGradient(const size_t rows, const size_t cols, const float prob) {
        Matrix<T> Yh(Shape{rows, 1}, prob), Y(Shape{rows, 1}, 1.0f);
        CrossEntropy<T> cost(cols);
        cost.Evaluate(Yh, Y);
        cost.CopyDeviceToHost();
        const Matrix<T>& dY = cost.GetGradient();
        EXPECT_EQ(dY.GetShape().rows, rows);
        EXPECT_EQ(dY.GetShape().cols, 1);
        for(size_t c = 0; c < rows; ++c) {
            const auto estimation = -(1.0 - prob);
            AssertEqual(dY(0, c), estimation);
        }
        return dY;
    }
};

TEST_F(CrossEntropyTest, BinaryCrossEntropy) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand(1, 4096);
        const auto cols = Rand(1, 32);
        const auto prob = Rand(0.001f, 0.999f);
        const auto cost_cpu = TestCrossEntropy<Cpu>(rows, cols, prob);
        const auto cost_cuda = TestCrossEntropy<Cuda>(rows, cols, prob);
        AssertEqual(cost_cpu, cost_cuda);
    }
}

TEST_F(CrossEntropyTest, BinaryCrossEntropyDerivative) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand(1, 4096);
        const auto cols = Rand(1, 32);
        const auto prob = Rand(0.001f, 0.999f);
        const auto dY_cpu = TestCrossEntropyGradient<Cpu>(rows, cols, prob);
        const auto dY_cuda = TestCrossEntropyGradient<Cuda>(rows, cols, prob);
        ASSERT_EQ(dY_cpu.GetShape().rows, dY_cuda.GetShape().rows);
        ASSERT_EQ(dY_cpu.GetShape().cols, dY_cuda.GetShape().cols);
        for(size_t c = 0; c < dY_cpu.GetShape().cols; ++c) {
            AssertEqual(dY_cpu(0, c), dY_cuda(0, c));
        }
    }
}

} //namespace cost
} //namespace nn
