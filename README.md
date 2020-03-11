
# Build Docker Container with CUDA Support

```
docker build -t cuda-dev -f cuda-dev.Dockerfile .
```

Test docker container:
```
docker run --gpus all -it --rm -v $PWD:/home/nn -w /home/nn cuda-dev nvidia-smi
```

# Build project in container

```
docker run --gpus all -it --rm -v $PWD:/home/nn -w /home/nn cuda-dev /bin/bash
bash ./download.sh
cd build
cmake -DUSE_CUDA=ON ..
cmake --build . -- -j 4
```

# MNIST example

```
./example/mnist/mnist
```

Network trains 100 epochs with accuracy 98.8%

# CIFAR-10 example

```
./example/cifar/cifar
```

https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

# Useful links

* https://www.fz-juelich.de/SharedDocs/Downloads/IAS/JSC/EN/slides/cuda/05-cuda-mm.pdf?__blob=publicationFile
* http://www.ncsa.illinois.edu/People/kindr/projects/hpca/files/NCSA_GPU_tutorial_d3.pdf
* https://www.tutorialspoint.com/cuda/cuda_matrix_multiplication.htm
* https://towardsdatascience.com/back-propagation-the-easy-way-part-3-cc1de33e8397
* https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
* http://luniak.io/cuda-neural-network-implementation-part-1/
* https://github.com/tiny-dnn/tiny-dnn
* https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver
* Adam Optimizer: https://github.com/maziarraissi/backprop/blob/master/CppSimpleNN.cpp
