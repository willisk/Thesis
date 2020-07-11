import os
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
tb = SummaryWriter()

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# plot_stats_mean, cat_cond_mean_
from utility import train, plot_decision_boundary,
from datasets import Dataset01, Dataset02

# %% utility


def train(net, data_loader, num_epochs=1, print_every=10):

    net.train()

    for epoch in range(1, num_epochs + 1):

        running_loss = 0.0

        for i, data in enumerate(data_loader, 1):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}

            net.optimizer.zero_grad()

            outputs = net(x)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            running_loss += loss.item()

        net.stop_statistics_tracking()

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, running_loss))
    print("Finished Training")

# %%


def plot_stats_mean(mean):
    cmap = 'Spectral'
    L = mean.shape[0]
    if L == 1:
        plt.scatter(mean[0], mean[1], c='k')
    else:
        mean = mean.T
        plt.scatter(mean[0], mean[1], c=np.arange(L),
                    cmap=cmap, edgecolors='k', alpha=0.5)


def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, class_count):
    batch_size = len(inputs)
    total = torch.zeros((batch_size, n_classes, n_features))
    for i, sample in enumerate(inputs):
        total[i, labels[i]] = sample

    N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)
    curr_class_f = class_count.float()
    class_count.add_(N_class)
    mean.mul_(curr_class_f / class_count)
    mean.add_(total.sum(axis=0) / class_count)


# %%
class CStatistics(nn.Module):

    def __init__(self, num_features, num_classes):
        super(CStatistics, self).__init__()
        shape = (num_classes, num_features)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('class_count',
                             torch.ones((num_classes, 1), dtype=torch.long))
        self.register_buffer('tracking', torch.ones(1, dtype=torch.bool))
        self.num_classes = num_classes
        self.num_features = num_features
        self.shape = shape

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.class_count.fill_(1)

    def forward(self, inputs, labels):
        if self.training and self.tracking:
            # N = self.class_count
            # shape = self.running_mean.shape
            # N = torch.zeros(shape[0])
            cat_cond_mean_(
                inputs, labels, self.num_classes, self.num_features,
                mean=self.running_mean,
                class_count=self.class_count)
            # self.running_mean = mean
            # self.running_mean = (
            #     self.running_mean * self.class_count + mean) / (N + self.class_count)
            # self.running_var = (self.running_var * N + std) / (N + 1)
        return inputs

    def stop_tracking(self):
        self.tracking = torch.zeros(1, dtype=torch.bool)


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
            n_out = layer_dims[i + 1]
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
                x = self.stats[i + 1](x, labels)
        return x

    def compile(self, lr=0.01):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def predict(self, x):
        net.eval()
        y = self.__call__(x)
        return torch.argmax(y, dim=-1)

    def stop_statistics_tracking(self):
        for m in self.modules():
            if isinstance(m, CStatistics):
                m.stop_tracking()


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
