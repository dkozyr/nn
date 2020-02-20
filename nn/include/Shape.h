#pragma once

#include <cstddef>

namespace nn {

struct Shape {
	size_t layers, rows, cols;

	Shape(size_t r, size_t c) : layers(1), rows(r), cols(c) {}
	Shape(size_t k, size_t r, size_t c) : layers(k), rows(r), cols(c) {}

	size_t Size() const { return layers * rows * cols; }
	size_t LayerSize() const { return rows * cols; }
	size_t RowSize() const { return cols; }
};

} //namespace nn
