import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
mnist_train = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST',
                                                train=True, download=False,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST',
                                               train=False, download=False,
                                               transform=transforms.ToTensor())
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_test[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
