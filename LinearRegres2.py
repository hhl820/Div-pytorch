""" 三个要素：
1. 训练数据
feature和对应的label
2. 损失函数
训练模型中，衡量价格预测值与真实值之间的误差
3. 优化算法
小批量随机梯度下降（mini-batch stochastic gradient descent)
每次迭代中，先随机先随机均匀采样⼀一个由固定数⽬目训练
数据样本所组成的⼩小批量量（mini-batch） ，然后求⼩小批量量中数
据样本的平均损失有关模型参数的导数（梯度），最后⽤用此结果与预先
设定的⼀一个正数的乘积作为模型参数在本次迭代的减小量。
"""
import torch
import numpy as np
import random
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
num_input = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0, 1, (num_input, num_examples)))
labels = true_w[0] * features[0, :] + true_w[1] * features[1, :] + true_b
print(labels.shape)
labels += torch.from_numpy(np.random.normal(0, 0.01), labels.size())





class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    
    def forward(self, features):
        labels = self.linear(features)
        return labels

model = LinearNet(2)

for epoch in range(30):
    loss = 0
    for X, y in data_iter(batch_size, features, labels):
        y_hat = model(features)
        loss += (y - y_hat)**2
    write.add_scalars('scalar/linear', loss, epoch)




