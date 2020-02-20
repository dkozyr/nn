#include "Common.h"
#include "cost/BinaryCrossEntropy.h"

namespace nn {
namespace cost {

constexpr float kEps = 1e-3;

class BinaryCrossEntropyTest: public ::testing::Test {
public:
    BinaryCrossEntropyTest() = default;

protected:
    template <ExecutorType T>
    static float TestBinaryCrossEntropy(size_t rows, float prob) {
        Matrix<T> Yh(Shape{rows, 1}, prob), Y(Shape{rows, 1}, 1.0f);
        BinaryCrossEntropy<T> cost;
        cost.Evaluate(Yh, Y);
        const auto estimation = -std::log(prob);
        AssertEqual(estimation, cost.GetLoss());
        return cost.GetLoss();
    }

    template <ExecutorType T>
    static Matrix<T> TestBinaryCrossEntropyGradient(size_t rows, float prob) {
        Matrix<T> Yh(Shape{rows, 1}, prob), Y(Shape{rows, 1}, 1.0f);
        BinaryCrossEntropy<T> cost;
        cost.Evaluate(Yh, Y);
        cost.CopyDeviceToHost();
        const Matrix<T>& dY = cost.GetGradient();
        EXPECT_EQ(dY.GetShape().rows, rows);
        EXPECT_EQ(dY.GetShape().cols, 1);
        for(size_t c = 0; c < rows; ++c) {
            const auto estimation = -(1.0 / prob);
            AssertEqual(dY(0, c), estimation);
        }
        return dY;
    }
};

TEST_F(BinaryCrossEntropyTest, BinaryCrossEntropy) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand(1, 4096);
        const auto prob = Rand(0.001f, 0.999f);
        const auto cost_cpu = TestBinaryCrossEntropy<Cpu>(rows, prob);
        const auto cost_cuda = TestBinaryCrossEntropy<Cuda>(rows, prob);
        AssertEqual(cost_cpu, cost_cuda);
    }
}

TEST_F(BinaryCrossEntropyTest, BinaryCrossEntropyDerivative) {
    for(auto begin = Clock::now(); Clock::now() - begin < 50ms; ) {
        const auto rows = Rand(1, 4096);
        const auto prob = Rand(0.001f, 0.999f);
        const auto dY_cpu = TestBinaryCrossEntropyGradient<Cpu>(rows, prob);
        const auto dY_cuda = TestBinaryCrossEntropyGradient<Cuda>(rows, prob);
        ASSERT_EQ(dY_cpu.GetShape().rows, dY_cuda.GetShape().rows);
        ASSERT_EQ(dY_cpu.GetShape().cols, dY_cuda.GetShape().cols);
        for(size_t c = 0; c < dY_cpu.GetShape().cols; ++c) {
            AssertEqual(dY_cpu(0, c), dY_cuda(0, c));
        }
    }
}

} //namespace cost
} //namespace nn
