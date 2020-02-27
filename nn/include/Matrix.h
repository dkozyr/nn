#pragma once

#include "Common.h"
#include "Shape.h"
#include "Utils.h"

namespace nn {

template <ExecutorType T = Cpu>
class Matrix {
public:
    explicit Matrix(const Shape& shape)
        : _max_size(shape.Size())
        , _shape(shape) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers == 0)) {
            throw "Wrong matrix shape";
        }
        AllocateMemory(0);
    }

    explicit Matrix(const Shape& shape, float value)
        : _max_size(shape.Size())
        , _shape(shape) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers == 0)) {
            throw "Wrong matrix shape";
        }
        AllocateMemory(value);
    }

    const Shape& GetShape() const {
        return _shape;
    }

    void Reshape(const Shape& shape) {
        if(shape.Size() > _max_size) {
            throw "Wrong reshaping parameters";
        }
        _shape = shape;
    }

    const Matrix<T> GetSubMatrix(const size_t row_offset, const size_t rows) const {
        if(row_offset + rows > _shape.rows) {
            throw "Wrong sub-matrix size requested";
        }
        const auto ptr_offset = row_offset * _shape.cols;
        return Matrix<T>(Shape{rows, _shape.cols}, _data_host + ptr_offset, _data_device + ptr_offset);
    }

    void Xavier() {
        const auto x = sqrt(6.0f / (_shape.rows + _shape.cols));
        const auto size = _shape.Size();
        for(size_t i = 0; i < size; ++i) {
            _data_host[i] = Rand(-x, +x);
        }
    }

    void XavierNormal() {
        const float sigma = sqrt(2.0f / (_shape.layers + _shape.rows + _shape.cols));
        const auto size = _shape.Size();
        for(size_t i = 0; i < size; ++i) {
            _data_host[i] = RandNormal(0.0, sigma);
        }
    }

    void SetZeroValue() {
        SetHostValue(0.0f);
    }

    void CopyHostToDevice() const {}
    void CopyDeviceToHost() const {}

    template <ExecutorType TFrom>
    void CopyHostData(const Matrix<TFrom>& from) const {
        const auto& shape = from.GetShape();
        if((_shape.rows != shape.rows) || (_shape.cols != shape.cols) || (_shape.layers != shape.layers)) {
            throw "CopyHostData: wrong size";
        }
        memcpy(_data_host, from.HostData(), _shape.Size() * sizeof(float));
    }

    float* HostData() const {
        return _data_host;
    }

    float* DeviceData() const {
        if(_data_device) {
            return _data_device;
        } else {
            throw "Cannot get device data pointer: memory isn't allocated";
        }
    }

    void Save(std::fstream& file) const {
        CopyDeviceToHost();
        file.write(reinterpret_cast<char*>(HostData()), _shape.Size() * sizeof(float));
    }

    void Load(std::fstream& file) {
        file.read(reinterpret_cast<char*>(HostData()), _shape.Size() * sizeof(float));
        CopyHostToDevice();
    }

    float& operator[](const size_t index) {
        return _data_host[index];
    }

    const float& operator[](const size_t index) const {
        return _data_host[index];
    }

    float& operator()(const size_t row, const size_t column) {
        return _data_host[row * _shape.cols + column];
    }

    const float& operator()(const size_t row, const size_t column) const {
        return _data_host[row * _shape.cols + column];
    }

    float& operator()(const size_t layer, const size_t row, const size_t column) {
        return _data_host[(layer * _shape.rows + row) * _shape.cols + column];
    }

    const float& operator()(const size_t layer, const size_t row, const size_t column) const {
        return _data_host[(layer * _shape.rows + row) * _shape.cols + column];
    }

    void Debug(size_t N = kMaxBatch) const {
        for(size_t r = 0; r < std::min(_shape.rows, N); ++r) {
            for(size_t c = 0; c < _shape.cols; ++c) {
                std::cout << _data_host[r * _shape.cols + c] << " ";
            }
            std::cout << std::endl;
        }
    }

    void DebugDevice(size_t max_rows = 1000) const {
        std::cout << "DebugDevice: NOT CUDA!" << std::endl;
    }

private:
    explicit Matrix(const Shape& shape, float* data_host, float* data_device)
        : _max_size(shape.Size())
        , _shape(shape)
        , _data_host(data_host)
        , _data_device(data_device) {
        if((_shape.rows == 0) || (_shape.cols == 0) || (_shape.layers == 0)) {
            throw "Wrong matrix shape";
        }
    }

    void AllocateMemory(float value) {
        AllocateHostMemory(value);
    }

    void AllocateHostMemory(float value) {
        const auto size = _shape.Size();
        _data_host = static_cast<float*>(malloc(size * sizeof(float)));
        _data_host_ptr = std::shared_ptr<float>(_data_host, [&](float* ptr){ free(ptr); });
        SetHostValue(value);
    }

    void SetHostValue(const float value) {
        std::fill_n(_data_host, _shape.Size(), value);
    }

private:
    const size_t _max_size;
    Shape _shape;

    float* _data_host = nullptr;
    std::shared_ptr<float> _data_host_ptr = nullptr;
    float* _data_device = nullptr;
    std::shared_ptr<float> _data_device_ptr = nullptr;
};

} //namespace nn
