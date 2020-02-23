#include "Common.h"
#include "Network.h"

namespace nn {

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

class NetworkTest: public ::testing::Test {
public:
    NetworkTest() = default;

protected:
    template <ExecutorType T = Cpu>
    static Matrix<T> ReadMnistImage(const std::string& path) {
        const auto data = ReadFileToString(path);
        const auto N = ReadUint32(data, 4);
        const auto W = ReadUint32(data, 8);
        const auto H = ReadUint32(data, 12);
        if((ReadUint32(data, 0) != 0x0803) || (W != 28) || (H != 28)) {
            throw "wrong format";
        }
        size_t offset = 16;

        Matrix<T> X(Shape{N, W * H});
        for(size_t n = 0; n < N; ++n) {
            for(size_t i = 0; i < (W * H); ++i, ++offset) {
                X(n, i) = static_cast<uint8_t>(data[offset]) / 255.0;
            }
        }
        X.CopyHostToDevice();
        return X;
    }

    template <ExecutorType T = Cpu>
    static Matrix<T> ReadMnistLabel(const std::string& path) {
        const auto data = ReadFileToString(path);
        const auto N = ReadUint32(data, 4);
        if((ReadUint32(data, 0) != 0x0801)) {
            throw "wrong format";
        }
        size_t offset = 8;

        Matrix<T> Y(Shape{N, 10});
        for(size_t n = 0; n < N; ++n, ++offset) {
            Y(n, static_cast<uint8_t>(data[offset])) = 1.0f;
        }
        Y.CopyHostToDevice();
        return Y;
    }
};

TEST_F(NetworkTest, Xor) {
    Matrix<Cuda> X(Shape{4, 2}), Y(Shape{4, 1});
    X(0, 0) = 0, X(0, 1) = 0, Y(0, 0) = 0;
    X(1, 0) = 0, X(1, 1) = 1, Y(1, 0) = 1;
    X(2, 0) = 1, X(2, 1) = 0, Y(2, 0) = 1;
    X(3, 0) = 1, X(3, 1) = 1, Y(3, 0) = 0;
    X.CopyHostToDevice();
    Y.CopyHostToDevice();
    
    Network<Cuda> net("Xor");
    net.AddLinearLayer(2, 4);
    net.AddTanh();
    net.AddDropout(0.1);
    net.AddLinearLayer(4, 1);
    net.AddSigmoid();
    net.AddBinaryCrossEntropy();

    const float learning_rate = 0.1;
    for(size_t epoch = 1; epoch <= 10000; ++epoch) {
        const float loss = net.Train(X, Y, learning_rate);
        if(epoch % 1000 == 0) {
            std::cout << "epoch: " << epoch << ", error: " << loss << ", ";
            std::cout << "Accuracy: " << net.Accuracy(X, Y) << std::endl;
        }
    }

    auto Yh = net.Predict(X);
    Yh.CopyDeviceToHost();
    Yh.Debug();
    ASSERT_LE(Yh[0], 0.05);
    ASSERT_GE(Yh[1], 0.95);
    ASSERT_GE(Yh[2], 0.95);
    ASSERT_LE(Yh[3], 0.05);
}

TEST_F(NetworkTest, Mnist) {
    if(!IsCudaEnabled()) {
        return; // on CPU it takes ~13 min, GPU ~25 sec
    }

    auto X_train = ReadMnistImage<Cuda>("../data/mnist/train-images-idx3-ubyte");
    auto Y_train = ReadMnistLabel<Cuda>("../data/mnist/train-labels-idx1-ubyte");

    auto X_cv = ReadMnistImage<Cuda>("../data/mnist/t10k-images-idx3-ubyte");
    auto Y_cv = ReadMnistLabel<Cuda>("../data/mnist/t10k-labels-idx1-ubyte");

    Network<Cuda> net("mnist");
    net.AddLinearLayer(28 * 28, 64);
    net.AddTanh();
    net.AddBatchNorm();
    net.AddDropout(0.3);
    net.AddLinearLayer(64, 64);
    net.AddTanh();
    net.AddDropout(0.2);
    net.AddLinearLayer(64, 10);
    net.AddSoftmax();
    net.AddCrossEntropy(10);

    const float learning_rate = 0.1;
    for(size_t epoch = 1; epoch <= 10; ++epoch) {
        const float loss = net.Train(X_train, Y_train, learning_rate, 16);
        std::cout << "epoch: " << epoch << ", error: " << loss;
        // std::cout << ", Accuracy train: " << net.Accuracy(X_train, Y_train);
        // std::cout << ", CV: " << net.Accuracy(X_cv, Y_cv);
        std::cout << std::endl;
    }
    const auto train_acc = net.Accuracy(X_train, Y_train);
    const auto cv_acc = net.Accuracy(X_cv, Y_cv);
    cout << "Accuracy train: " << train_acc << ", CV: " << cv_acc << endl;
    ASSERT_GE(train_acc, 0.96);
    ASSERT_GE(cv_acc, 0.95); // need Conv-layer and more epochs to get better accuracy
}

TEST_F(NetworkTest, MnistConv) {
    if(!IsCudaEnabled()) {
        return; // on CPU it takes ~17 min, GPU ~30 sec
    }

    auto X_train = ReadMnistImage<Cuda>("../data/mnist/train-images-idx3-ubyte");
    auto Y_train = ReadMnistLabel<Cuda>("../data/mnist/train-labels-idx1-ubyte");

    auto X_cv = ReadMnistImage<Cuda>("../data/mnist/t10k-images-idx3-ubyte");
    auto Y_cv = ReadMnistLabel<Cuda>("../data/mnist/t10k-labels-idx1-ubyte");

    Network<Cuda> net("mnist");
    net.AddConv(Shape{28,28}, Shape{2,2});
    net.AddReLu();
    net.AddDropout(0.2);

    net.AddLinearLayer(27*27, 100);
    net.AddTanh();
    net.AddDropout(0.1);

    net.AddLinearLayer(100, 10);
    net.AddSoftmax();
    net.AddCrossEntropy(10);

    const float learning_rate = 0.1;
    for(size_t epoch = 1; epoch <= 10; ++epoch) {
        const float loss = net.Train(X_train, Y_train, learning_rate, 16);
        std::cout << "epoch: " << epoch << ", error: " << loss;
        // std::cout << ", Accuracy train: " << net.Accuracy(X_train, Y_train);
        // std::cout << ", CV: " << net.Accuracy(X_cv, Y_cv);
        std::cout << std::endl;
    }
    const auto train_acc = net.Accuracy(X_train, Y_train);
    const auto cv_acc = net.Accuracy(X_cv, Y_cv);
    cout << "Accuracy train: " << train_acc << ", CV: " << cv_acc << endl;
    ASSERT_GE(train_acc, 0.975);
    ASSERT_GE(cv_acc, 0.97);
}

TEST_F(NetworkTest, MnistConv3D) {
    if(!IsCudaEnabled()) {
        return; // on CPU it takes ~17 min, GPU ~60 sec
    }

    auto X_train = ReadMnistImage<Cuda>("../data/mnist/train-images-idx3-ubyte");
    auto Y_train = ReadMnistLabel<Cuda>("../data/mnist/train-labels-idx1-ubyte");

    auto X_cv = ReadMnistImage<Cuda>("../data/mnist/t10k-images-idx3-ubyte");
    auto Y_cv = ReadMnistLabel<Cuda>("../data/mnist/t10k-labels-idx1-ubyte");

    Network<Cuda> net("mnist");
    net.AddConv3D(Shape{28,28}, Shape{4,2,2});
    net.AddReLu();
    net.AddDropout(0.2);

    net.AddLinearLayer(4*27*27, 100);
    net.AddTanh();
    net.AddDropout(0.1);

    net.AddLinearLayer(100, 10);
    net.AddSoftmax();
    net.AddCrossEntropy(10);

    const float learning_rate = 0.1;
    for(size_t epoch = 1; epoch <= 10; ++epoch) {
        const float loss = net.Train(X_train, Y_train, learning_rate, 16);
        std::cout << "epoch: " << epoch << ", error: " << loss;
        // std::cout << ", Accuracy train: " << net.Accuracy(X_train, Y_train);
        // std::cout << ", CV: " << net.Accuracy(X_cv, Y_cv);
        std::cout << std::endl;
    }
    const auto train_acc = net.Accuracy(X_train, Y_train);
    const auto cv_acc = net.Accuracy(X_cv, Y_cv);
    cout << "Accuracy train: " << train_acc << ", CV: " << cv_acc << endl;
    ASSERT_GE(train_acc, 0.98);
    ASSERT_GE(cv_acc, 0.97);
}

} //namespace nn
