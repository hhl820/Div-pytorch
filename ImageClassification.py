import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
# mnist_train = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST',
#                                                 train=True, download=False,
#                                                 transform=transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST',
#                                                train=False, download=False,
#                                                transform=transforms.ToTensor())
# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_test[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 2
# train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
#                                          shuffle=True, num_workers=num_workers)
# test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
#                                         shuffle=True, num_workers=num_workers)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10
class Linearnet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Linearnet, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y
net = Linearnet(num_inputs, num_outputs)

torch.nn.init.normal_(net.linear.weight)
torch.nn.init.normal_(net.linear.bias)

loss = torch.nn.CrossEntropyLoss()
optimzer = optim.SGD(net.parameters(), lr=0.1)
for epoch in range(50):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimzer.zero_grad()
        l.backward()
        optimzer.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    print('epoch: %d, loss: %f, acc: %f'%(epoch+1, train_l_sum/n, train_acc_sum/n))






