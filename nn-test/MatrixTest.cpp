#include "Common.h"
#include "Matrix.h"

namespace nn {

class MatrixTest: public ::testing::Test {
public:
    MatrixTest() = default;

protected:
};

TEST_F(MatrixTest, Simple) {
    Matrix<Cpu> matrix(Shape{1,1});
}

TEST_F(MatrixTest, Cuda) {
    Matrix<Cuda> matrix(Shape{1,1});
}

} //namespace nn
