#if USE_CUDA

#include "cuda/Common.h"
#include "Matrix.h"

namespace nn {

template <>
void Matrix<Cuda, float>::CopyHostToDevice() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_device_ptr.get(), _data_host_ptr.get(), _shape.Size() * sizeof(float), cudaMemcpyHostToDevice);
        Exception::ThrowOnError("Cannot copy data from host to CUDA device");
    }
}

template <>
void Matrix<Cuda, uint32_t>::CopyHostToDevice() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_device_ptr.get(), _data_host_ptr.get(), _shape.Size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        Exception::ThrowOnError("Cannot copy data from host to CUDA device");
    }
}

template <>
void Matrix<Cuda, float>::CopyDeviceToHost() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_host_ptr.get(), _data_device_ptr.get(), _shape.Size() * sizeof(float), cudaMemcpyDeviceToHost);
        Exception::ThrowOnError("Cannot copy data from CUDA device to host");
    }
}

template <>
void Matrix<Cuda, uint32_t>::CopyDeviceToHost() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_host_ptr.get(), _data_device_ptr.get(), _shape.Size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        Exception::ThrowOnError("Cannot copy data from CUDA device to host");
    }
}

template <>
void Matrix<Cuda, float>::AllocateMemory(float value) {
    AllocateHostMemory(value);
    if(!_data_device_ptr) {
        cudaMalloc(&_data_device, _shape.Size() * sizeof(float));
        Exception::ThrowOnError("Matrix: Cannot allocate CUDA memory");

        _data_device_ptr = std::shared_ptr<float>(_data_device, [this](float* ptr){
            cudaFree(ptr);
        });

        CopyHostToDevice();
    }
}

template <>
void Matrix<Cuda, uint32_t>::AllocateMemory(uint32_t value) {
    AllocateHostMemory(value);
    if(!_data_device_ptr) {
        cudaMalloc(&_data_device, _shape.Size() * sizeof(uint32_t));
        Exception::ThrowOnError("Matrix: Cannot allocate CUDA memory");

        _data_device_ptr = std::shared_ptr<uint32_t>(_data_device, [this](uint32_t* ptr){
            cudaFree(ptr);
        });

        CopyHostToDevice();
    }
}

template <>
void Matrix<Cuda, float>::SetZeroValue() {
    cudaMemset(_data_device, 0, _shape.Size() * sizeof(float));
    Exception::ThrowOnError("Matrix: Cannot SetValue on CUDA");
}

template <>
void Matrix<Cuda, uint32_t>::SetZeroValue() {
    cudaMemset(_data_device, 0, _shape.Size() * sizeof(uint32_t));
    Exception::ThrowOnError("Matrix: Cannot SetValue on CUDA");
}

template <>
void Matrix<Cuda, float>::DebugDevice(size_t max_rows) const {
    if(_data_device) {
        cudaMemcpy(_data_host, _data_device, _shape.Size() * sizeof(float), cudaMemcpyDeviceToHost);

        // fist layer only
        for(size_t r = 0, i = 0; (r < this->_shape.rows) && (r < max_rows); ++r) {
            for(size_t c = 0; (c < this->_shape.cols) && (c < max_rows); ++c, ++i) {
                std::cout << _data_host[i] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "DebugDevice: memory is not allocated" << std::endl;
    }
}

} //namespace nn

#endif //USE_CUDA
