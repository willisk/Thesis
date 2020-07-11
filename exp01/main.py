# import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import Dataset01, Dataset02
from utility import train, plot_decision_boundary, plot_stats_mean, cat_cond_mean


# class Statistics(nn.Module):

#     def __init__(self, num_features):
#         super(Statistics, self).__init__()
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))
#         self.register_buffer('num_batches_tracked',
#                              torch.tensor(0, dtype=torch.long))

#     def reset_parameters(self):
#         self.running_mean.zero_()
#         self.running_var.fill_(1)
#         self.num_batches_tracked.zero_()

#     def forward(self, inputs):
#         if self.training:
#             N = self.num_batches_tracked
#             std, mean = torch.std_mean(inputs, dim=0)
#             self.running_mean = (self.running_mean * N + mean) / (N + 1)
#             self.running_var = (self.running_var * N + std) / (N + 1)
#             self.num_batches_tracked += 1
#         return inputs


class CStatistics(nn.Module):

    def __init__(self, num_features, num_classes):
        super(CStatistics, self).__init__()
        shape = (num_classes, num_features)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('num_classes_tracked',
                             torch.zeros(num_classes, dtype=torch.long))
        self.num_classes = num_classes
        self.num_features = num_features
        self.shape = shape

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_classes_tracked.zero_()

    def forward(self, inputs, labels):
        if self.training:
            # N = self.num_classes_tracked
            # shape = self.running_mean.shape
            # N = torch.zeros(shape[0])
            batch_size = len(inputs)

            mean, N = cat_cond_mean(
                inputs, labels, self.num_classes, self.num_features)
            self.running_mean = mean
            # self.running_mean = (self.running_mean * N + mean) / (N + 1)
            # self.running_var = (self.running_var * N + std) / (N + 1)
            self.num_batches_tracked = N
        return inputs


class Net01(nn.Module):

    def __init__(self, layer_dims):
        super(Net01, self).__init__()
        self.fc = nn.ModuleList()
        self.stats = nn.ModuleList()

        n_classes = layer_dims[-1]
        n_in = layer_dims[0]
        self.stats.append(CStatistics(n_in, n_classes))

        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i+1]
            layer = nn.Linear(n_in, n_out)
            self.fc.append(layer)
            self.stats.append(CStatistics(n_out, n_classes))

    def forward(self, data):

        if self.training:
            x = data['inputs']
            labels = data['labels']
        else:
            x = data

        if self.training:
            x = self.stats[0](x, labels)
        L = len(self.fc)
        for i in range(L):
            x = self.fc[i](x)
            if i < L - 1:
                x = F.relu(x)
            if self.training:
                x = self.stats[i+1](x, labels)
        return x

    def compile(self, lr=0.01):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def predict(self, x):
        net.eval()
        y = self.__call__(x)
        return torch.argmax(y, dim=-1)


training_params = {'num_epochs': 100,
                   'print_every': 20}
dataset_params = {
    'n_samples': 100,
    # 'noise': 0.1,
    # 'factor': 0.5
}
dataset_gen_params = {'batch_size': 64,
                      'shuffle': True}

net = Net01([2, 8, 6, 3])
net.compile(lr=0.01)
dataset = Dataset01(**dataset_params)
# inputs, labels = next(iter(data_loader))
train(net, dataset.generator(**dataset_gen_params), **training_params)
plot_decision_boundary(dataset, net)
stats = [m for m in net.modules() if isinstance(m, CStatistics)]
plot_stats_mean(stats[0].running_mean.data.detach())
plt.show()
