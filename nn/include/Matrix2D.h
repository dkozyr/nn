#pragma once

#include "Matrix.h"

namespace nn {

template <ExecutorType T = Cpu>
class Matrix2D : public Matrix {
public:
    explicit Matrix2D(const Shape& shape)
        : Matrix<T>(shape) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers != 1)) {
            throw "Wrong matrix shape";
        }
    }

    explicit Matrix2D(const Shape& shape, float value)
        : Matrix<T>(shape) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers != 1)) {
            throw "Wrong matrix shape";
        }
    }

private:
    explicit Matrix2D(const Shape& shape, float* data_host, float* data_device)
        : Matrix<T>(shape, data_host, data_device) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers != 1)) {
            throw "Wrong matrix shape";
        }
    }
};

} //namespace nn
