#include <fstream>

#include "Network.h"

using namespace std;
using namespace nn;

void ReadCifarData(const std::string& path, Matrix<Cuda>& X, Matrix<Cuda>& Y, size_t idx);
inline std::string ReadFileToString(const std::string& path);

void Run() {
    Matrix<Cuda> X_train(Shape{50000, 3*32*32}), Y_train(Shape{50000, 10});
    for(size_t i = 1; i <= 5; ++i) {
        const std::string path = "../data/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin";
        ReadCifarData(path, X_train, Y_train, (i - 1) * 10000);
    }
    cout << "Train images: " << X_train.GetShape().rows << endl;

    Matrix<Cuda> X_cv(Shape{10000, 3*32*32}), Y_cv(Shape{10000, 10});
    ReadCifarData("../data/cifar-10-batches-bin/test_batch.bin", X_cv, Y_cv, 0);
    cout << "CV images: " << X_cv.GetShape().rows << endl;

    Network<Cuda> net("cifar");

    net.AddConv3D(Shape{3,32,32}, Shape{16,5,5});
    net.AddReLu();
    net.AddMaxPool(Shape{16,28,28}, Shape{2,2});
    net.AddDropout(0.2);

    net.AddConv3D(Shape{16,27,27}, Shape{32,5,5});
    net.AddReLu();
    net.AddMaxPool(Shape{32,23,23}, Shape{2,2});
    net.AddDropout(0.2);

    net.AddLinearLayer(32*22*22, 256);
    net.AddReLu();
    net.AddDropout(0.2);

    net.AddLinearLayer(256, 10);
    net.AddSoftmax();
    net.AddCrossEntropy(10);

    const float learning_rate = 0.1;
    const size_t batch = 64;
    for(size_t epoch = 1; epoch <= 50; ++epoch) {
        const float loss = net.Train(X_train, Y_train, learning_rate, batch);
        cout << "epoch: " << epoch << ", error: " << loss;
        if(epoch % 10 == 0) {
            cout << ", Accuracy train: " << net.Accuracy(X_train, Y_train);
            cout << ", CV: " << net.Accuracy(X_cv, Y_cv);
        }
        cout << endl;
        // net.Save("cifar.dat");
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

void ReadCifarData(const std::string& path, Matrix<Cuda>& X, Matrix<Cuda>& Y, size_t idx) {
    const auto data = ReadFileToString(path);
    const auto N = data.size() / (3*32*32 + 1);
    if((N * (3*32*32 + 1)) != data.size()) {
        throw "wrong cifar data file";
    }
    size_t offset = 0;
    for(size_t n = 0; n < N; ++n, ++idx) {
        Y(idx, data[offset]) = 1.0;
        ++offset;
        for(size_t i = 0; i < 3*32*32; ++i, ++offset) {
            X(idx, i) = static_cast<uint8_t>(data[offset]) / 255.0;
        }
    }
    X.CopyHostToDevice();
    Y.CopyHostToDevice();
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
