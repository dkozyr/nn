#if USE_CUDA

#include "cuda/Common.h"
#include "Matrix.h"

namespace nn {

template <>
void Matrix<Cuda>::CopyHostToDevice() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_device_ptr.get(), _data_host_ptr.get(), _shape.Size() * sizeof(float), cudaMemcpyHostToDevice);
        Exception::ThrowOnError("Cannot copy data from host to CUDA device");
    }
}

template <>
void Matrix<Cuda>::CopyDeviceToHost() const {
    if(!_data_device_ptr || !_data_host_ptr) {
        throw Exception("Cannot copy data: memory isn't allocated");
    } else {
        cudaMemcpy(_data_host_ptr.get(), _data_device_ptr.get(), _shape.Size() * sizeof(float), cudaMemcpyDeviceToHost);
        Exception::ThrowOnError("Cannot copy data from CUDA device to host");
    }
}

template <>
void Matrix<Cuda>::AllocateMemory(float value) {
    AllocateHostMemory(value);

    if(!_data_device_ptr) {
        cudaMalloc(&_data_device, _shape.Size() * sizeof(float));
        // cout << "Matrix " << _shape.layers << "x" << _shape.rows << "x" << _shape.cols
        //      << " -> size: " << _shape.Size() << endl;
        Exception::ThrowOnError("Matrix: Cannot allocate CUDA memory");

        _data_device_ptr = std::shared_ptr<float>(_data_device, [this](float* ptr){
            cudaFree(ptr);
        });

        CopyHostToDevice();
    }
}

template <>
void Matrix<Cuda>::SetZeroValue() {
    cudaMemset(_data_device, 0, _shape.Size() * sizeof(int));
    Exception::ThrowOnError("Matrix: Cannot SetValue on CUDA");
}


template <>
void Matrix<Cuda>::DebugDevice(size_t max_rows) const {
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
