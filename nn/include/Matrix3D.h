#pragma once

#include "Matrix.h"

namespace nn {

template <ExecutorType T = Cpu>
using Matrix3D = Matrix<T>;

} //namespace nn
