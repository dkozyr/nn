#include <fstream>

#include "Network.h"

using namespace std;
using namespace nn;

Matrix<Cuda> ReadMnistImage(const std::string& path);
Matrix<Cuda> ReadMnistLabel(const std::string& path);
inline std::string ReadFileToString(const std::string& path);

void Run() {
    auto X_train = ReadMnistImage("../data/mnist/train-images-idx3-ubyte");
    auto Y_train = ReadMnistLabel("../data/mnist/train-labels-idx1-ubyte");
    cout << "Train images: " << X_train.GetShape().rows << endl;

    auto X_cv = ReadMnistImage("../data/mnist/t10k-images-idx3-ubyte");
    auto Y_cv = ReadMnistLabel("../data/mnist/t10k-labels-idx1-ubyte");
    cout << "CV images: " << X_cv.GetShape().rows << endl;

    Network<Cuda> net("mnist");

    net.AddConv3D(Shape{28,28}, Shape{32,5,5});
    net.AddReLu();
    net.AddMaxPool(Shape{32,24,24}, Shape{2,2});
    net.AddDropout(0.2);

    net.AddConv3D(Shape{32,23,23}, Shape{64,5,5});
    net.AddReLu();
    net.AddMaxPool(Shape{64,19,19}, Shape{2,2});
    net.AddDropout(0.2);

    net.AddLinearLayer(64*18*18, 512);
    net.AddReLu();
    net.AddDropout(0.1);

    net.AddLinearLayer(512, 10);
    net.AddSoftmax();
    net.AddCrossEntropy(10);

    const float learning_rate = 0.1;
    for(size_t epoch = 1; epoch <= 20; ++epoch) {
        const float loss = net.Train(X_train, Y_train, learning_rate, 16);
        cout << "epoch: " << epoch << ", error: " << loss;
        if(epoch % 10 == 0) {
            cout << ", Accuracy train: " << net.Accuracy(X_train, Y_train);
            cout << ", CV: " << net.Accuracy(X_cv, Y_cv);
        }
        cout << endl;
    }
}

int main(int argc, char**argv) {
    cout << "Cuda is " << (IsCudaEnabled() ? "ON" : "OFF") << endl;
    try {
        Run();
    } catch(const char* error) {
        cout << "Error: " << error << endl;
    }
    return 0;
}

Matrix<Cuda> ReadMnistImage(const std::string& path) {
    const auto data = ReadFileToString(path);
    const auto N = ReadUint32(data, 4);
    const auto W = ReadUint32(data, 8);
    const auto H = ReadUint32(data, 12);
    if((ReadUint32(data, 0) != 0x0803) || (W != 28) || (H != 28)) {
        throw "wrong mnist file format";
    }
    size_t offset = 16;

    Matrix<Cuda> X(Shape{N, W * H});
    for(size_t n = 0; n < N; ++n) {
        for(size_t i = 0; i < (W * H); ++i, ++offset) {
            X(n, i) = static_cast<uint8_t>(data[offset]) / 255.0;
        }
    }
    X.CopyHostToDevice();
    return X;
}

Matrix<Cuda> ReadMnistLabel(const std::string& path) {
    const auto data = ReadFileToString(path);
    const auto N = ReadUint32(data, 4);
    if((ReadUint32(data, 0) != 0x0801)) {
        throw "wrong mnist file format";
    }
    size_t offset = 8;

    Matrix<Cuda> Y(Shape{N, 10});
    for(size_t n = 0; n < N; ++n, ++offset) {
        Y(n, static_cast<uint8_t>(data[offset])) = 1.0f;
    }
    Y.CopyHostToDevice();
    return Y;
}

inline std::string ReadFileToString(const std::string& path) {
    std::ifstream infile(path.c_str());

    infile.seekg(0, std::ios::end);
    const auto file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::string raw(file_size, '\0');
    infile.read(raw.data(), raw.size());
    infile.close();
    return raw;
}
