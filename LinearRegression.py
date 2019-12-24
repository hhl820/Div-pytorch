import torch.utils.data as Data
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
num_Feature = 2
num_Sample = 1000
features = torch.tensor(np.random.normal(0, 1,
                        (num_Sample, num_Feature)), dtype=torch.float32)
true_w = [2, -3.4]
true_b = 4.2
labels = true_w[0]*features[:, 0] + true_w[1]*features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01,
                       size=labels.size()), dtype=torch.float32)
features = features.float()
labels = labels.float()
dataset = Data.TensorDataset(features, labels)
batch_size = 10
dataiter = Data.DataLoader(dataset, batch_size, shuffle=True)


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_Feature)
print(net)
net1 = nn.Sequential()
net1.add_module('linear', nn.Linear(num_Feature, 1))
init.normal_(net1[0].weight, mean=0, std=0.01)
init.constant_(net1[0].bias, val=0)

loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)
# print(optimizer)
LOSS = 0
num_epochs = 10
for i in range(1, num_epochs+1):
    for X, y in dataiter:
        output = net(X)
        LOSS = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()
    print('epoch: %d, loss: %f' % (i, LOSS))
