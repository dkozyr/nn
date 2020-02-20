#!/bin/sh

mkdir -p ./data
mkdir -p ./data/mnist

curl -sS http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > ./data/mnist/train-images-idx3-ubyte
curl -sS http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > ./data/mnist/train-labels-idx1-ubyte
curl -sS http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > ./data/mnist/t10k-images-idx3-ubyte
curl -sS http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > ./data/mnist/t10k-labels-idx1-ubyte

cd ./data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xf cifar-10-binary.tar.gz
rm cifar-10-binary.tar.gz

wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar xf cifar-100-binary.tar.gz
rm cifar-100-binary.tar.gz
cd ..
