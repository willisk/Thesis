import os
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# plot_stats_mean, cat_cond_mean_
from utility import train, plot_decision_boundary
from datasets import Dataset2D


comment = ""
tb = SummaryWriter(comment=comment)

# %% utility


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum()


def train(net, data_loader, num_epochs=1, print_every=10):
    "Training Loop"

    net.train()
    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for data in data_loader:
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}

            net.optimizer.zero_grad()

            outputs = net(x)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            batch_size = len(inputs)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_correct += count_correct(outputs, labels)

        net.stop_statistics_tracking()

        accuracy = total_correct / total_count
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Accuracy', accuracy, epoch)

        # for i, layer in enumerate(net.fc, 1):
        #     tb.add_histogram("fc{}.bias".format(i), layer.bias, epoch)
        #     tb.add_histogram("fc{}.weight".format(i), layer.weight, epoch)
        #     tb.add_histogram("fc{}.weight.grad".format(i),
        #                      layer.weight.grad, epoch)

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, total_loss))
    print("Finished Training")

# %%


def plot_stats_mean(stats):
    cmap = 'Spectral'
    mean = stats["running_mean"]

    num_classes = mean.shape[0]
    mean = mean.T
    if num_classes == 1:    # no class_conditional
        plt.scatter(mean[0], mean[1], c='k', alpha=0.4)
    else:
        plt.scatter(mean[0], mean[1], c=np.arange(num_classes),
                    cmap=cmap, edgecolors='k', alpha=0.4)


def plot_invert(x, labels):
    cmap = 'Spectral'
    x = x.detach().T
    plt.scatter(x[0], x[1], c=labels,
                cmap=cmap, edgecolors='k', alpha=0.4,
                marker='^')


def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, cc, class_conditional=True):

    if not class_conditional:
        total = inputs.sum(dim=0)
        N_class = len(inputs)
    else:
        total = torch.zeros((n_classes, n_features))
        total.index_add_(0, labels, inputs.detach())
        N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)

    cc_f = cc.detach()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)


# %%
class CStatistics(nn.Module):

    def __init__(self, num_features, num_classes, class_conditional=True):
        super(CStatistics, self).__init__()
        if not class_conditional:
            num_classes = 1
        shape = (num_classes, num_features)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('class_count',
                             torch.ones((num_classes, 1), dtype=torch.long))
        self.register_buffer('tracking', torch.ones(1, dtype=torch.bool))
        self.num_classes = num_classes
        self.num_features = num_features
        self.shape = shape
        self.class_conditional = class_conditional

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.class_count.fill_(1)

    def forward(self, inputs, labels):
        if self.training:
            if self.tracking:
                cat_cond_mean_(
                    inputs, labels, self.num_classes, self.num_features,
                    mean=self.running_mean,
                    cc=self.class_count,
                    class_conditional=self.class_conditional)
        else:
            # print("rm req grad: " + str(self.running_mean.requires_grad))
            means = self.running_mean[labels]
            # print("m req grad: " + str(means.requires_grad))
            self.regularization = (inputs - means).norm(2).sum()
        return inputs

    def stop_tracking(self):
        self.tracking = torch.zeros(1, dtype=torch.bool)


class Net01(nn.Module):

    def __init__(self, layer_dims, class_conditional=True):
        super(Net01, self).__init__()

        self.fc = nn.ModuleList()
        self.stats = nn.ModuleList()

        n_classes = layer_dims[-1]
        n_in = layer_dims[0]
        self.stats.append(CStatistics(
            n_in, n_classes, class_conditional=class_conditional))

        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            layer = nn.Linear(n_in, n_out)
            self.fc.append(layer)
            self.stats.append(CStatistics(n_out, n_classes,
                                          class_conditional=class_conditional))

        self.class_conditional = class_conditional

    def forward(self, data):

        use_stats = isinstance(data, dict) and 'labels' in data

        if use_stats:
            x = data['inputs']
            labels = data['labels']
        else:
            x = data

        L = len(self.fc)

        for i in range(L + 1):
            if use_stats:
                x = self.stats[i](x, labels)
            if i < L:
                x = self.fc[i](x)
            if i < L - 1:
                x = F.relu(x)
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

    def collect_stats(self):
        stat_vars = ['running_mean', 'running_var', 'class_count']
        stats = []
        for m in self.stats:
            stat = {}
            for s in stat_vars:
                stat[s] = getattr(m, s).data
            stats.append(stat)
        return stats

    def invert(self):
        self.inverting = True


def deep_inversion(net, labels, epochs=3):

    num_features = 2

    net.eval()

    bs = len(labels)
    inputs = torch.randn((bs, num_features), requires_grad=True)

    for epoch in range(1, epochs + 1):

        net.optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = net(data)
        loss = net.criterion(outputs, labels)

        reg_loss = sum([s.regularization for s in net.stats])

        loss = loss + reg_loss

        loss.backward()

        net.optimizer.step()

    return inputs


training_params = {'num_epochs': 100,
                   'print_every': 20}
dataset_params = {
    'n_samples': 100,
    # 'noise': 0.1,
    # 'factor': 0.5
}
dataset_gen_params = {'batch_size': 64,
                      'shuffle': True}

net = Net01([2, 8, 6, 3], class_conditional=True)
net.compile(lr=0.01)
dataset = Dataset2D(type=0, **dataset_params)
data_loader = dataset.generator(**dataset_gen_params)
# inputs, labels = next(iter(data_loader))
train(net, data_loader, **training_params)
plot_decision_boundary(dataset, net)
stats = net.collect_stats()
plot_stats_mean(stats[0])
target_labels = torch.arange(3)
# invert = deep_inversion(net, target_labels)
# plot_invert(invert, target_labels)
plt.show()

tb.close()
