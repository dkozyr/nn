# Neural Network C++ library with CUDA support

Simple c++ implementation for neural networks with CUDA support.

It was developed with purpose to learn deeply neural networks training process:
- activations: sigmoid, tanh, relu, softmax
- layers: linear, convolutional, max-pool
- backpropagation algorithm
- dropout
- batch normalization

TODO:
- fix dropout using CUDA random generators implementation
- adam optimizer
- cifar-100

## Build project (Linux / Docker contanier)

```
cd build
cmake -DUSE_CUDA=ON ..
cmake --build . -- -j 4
```

## Download MNIST and CIFAR data

```
bash ./download.sh
```

## MNIST example

Convolutional neural network architecture:
```
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
```

Network trains 20 epochs with accuracy 99.0%

## CIFAR-10 example

```
./example/cifar/cifar
```

https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

## Build Linux-based docker container with CUDA support

```
docker build -t cuda-dev -f cuda-dev.Dockerfile .
```

Test it:
```
docker run --gpus all cuda-dev nvidia-smi
```

Run container:
```
docker run --gpus all -it --rm -v $PWD:/home/nn -w /home/nn cuda-dev /bin/bash
```

## Useful links

tiny-dnn is a good implementation of deep learning algorithms, but I couldn't get it work with CUDA:
* https://github.com/tiny-dnn/tiny-dnn

CUDA Neural Network Implementation:
* http://luniak.io/cuda-neural-network-implementation-part-1/

Back Propagation (and Python example):
* https://towardsdatascience.com/back-propagation-the-easy-way-part-3-cc1de33e8397

Matrix multiplication with Cuda:
* https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cuda/05-cuda-mm.pdf?__blob=publicationFile
* http://www.ncsa.illinois.edu/People/kindr/projects/hpca/files/NCSA_GPU_tutorial_d3.pdf
* https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/

CUDA Compatibility:
* https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver

Adam Optimizer:
* https://github.com/maziarraissi/backprop/blob/master/CppSimpleNN.cpp
