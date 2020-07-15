import os
import sys
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    batch_total = 1

    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for batch_i, data in enumerate(data_loader, batch_total):
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

            if not net.bypass_stats_layer:
                for i, layer in enumerate(net.stats, 1):
                    for j, feature in enumerate(layer.running_mean):
                        for m, f in enumerate(feature):
                            tb.add_scalar("stats%i.class%i.f%i" % (i, j, m + 1),
                                          feature[m], batch_i)

        batch_total = batch_i + 1

        # net.stop_statistics_tracking()

        accuracy = total_correct / total_count
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Accuracy', accuracy, epoch)

        # for i, layer in enumerate(net.fc, 1):
        #     tb.add_histogram("fc{}.bias".format(i), layer.bias, epoch)

        # for i, layer in enumerate(net.fc, 1):
        #     tb.add_histogram("fc{}.bias".format(i), layer.bias, epoch)
        #     tb.add_histogram("fc{}.weight".format(i), layer.weight, epoch)
        #     tb.add_histogram("fc{}.weight.grad".format(i),
        #                      layer.weight.grad, epoch)

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, total_loss))
    print("Finished Training")


def learn_stats(net, data_loader, num_epochs=1):

    net.eval()
    net.start_tracking_stats()

    batch_total = 1

    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for batch_i, data in enumerate(data_loader, batch_total):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            net(x)

            # for i, layer in enumerate(net.stats, 1):
            #     for j, feature in enumerate(layer.running_mean):
            #         for m, f in enumerate(feature):
            #             tb.add_scalar("stats%i.class%i.f%i" % (i, j, m + 1),
            #                           feature[m], batch_i)

        batch_total = batch_i + 1

    net.stop_tracking_stats()
    print("Finished Tracking Stats")

# %%


def plot_stats_mean(stats):
    cmap = 'Spectral'
    mean = stats["running_mean"]

    num_classes = mean.shape[0]
    mean = mean.T
    if num_classes == 1:    # no class_conditional
        c = 'k'
    else:
        c = np.arange(num_classes)
    plt.scatter(mean[0], mean[1], c=c,
                cmap=cmap, edgecolors='k', alpha=0.4,
                marker='^')


def plot_invert(x, labels):
    cmap = 'Spectral'
    x = x.detach().T
    plt.scatter(x[0], x[1], c=labels,
                cmap=cmap, edgecolors='k',
                marker='x',
                alpha=1)


def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, cc, class_conditional=True):

    if not class_conditional:
        total = inputs.sum(dim=0)
        N_class = len(inputs)
    else:
        total = torch.zeros((n_classes, n_features))
        total.index_add_(0, labels, inputs.detach())
        N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)

    cc_f = cc.float()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)


# %%


class Net01(nn.Module):

    def __init__(self, layer_dims):
        super(Net01, self).__init__()

        self.fc = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            n_in = layer_dims[i]
            n_out = layer_dims[i + 1]
            layer = nn.Linear(n_in, n_out)
            self.fc.append(layer)

    def forward(self, data):

        L = len(self.fc)

        for i in range(L):
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


class CStatistics(nn.Module):

    def __init__(self, num_features, num_classes, class_conditional=True,
                 unbiased_count=False):
        super(CStatistics, self).__init__()
        if not class_conditional:
            num_classes = 1
        if unbiased_count:
            cc_init = 1
        else:
            cc_init = 0
        shape = (num_classes, num_features)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('class_count',
                             torch.full((num_classes, 1),
                                        cc_init,
                                        dtype=torch.long))
        self.num_classes = num_classes
        self.num_features = num_features
        self.shape = shape
        self.class_conditional = class_conditional
        self.tracking_stats = False

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.class_count.fill_(1)

    def forward(self, inputs, labels):
        if self.tracking_stats:
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


class StatsHook():

    def __init__(self, module, num_features, num_classes,
                 input_stats=False, class_conditional=True, unbiased_count=False):
        self.hook = module.register_forward_hook(self.hook_fn)

        self.input_stats = input_stats
        self.num_classes = num_classes
        self.num_features = num_features
        self.class_conditional = class_conditional
        self.tracking_stats = False

        if not class_conditional:
            num_classes = 1
        if unbiased_count:
            cc_init = 1
        else:
            cc_init = 0
        shape = (num_classes, num_features)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))
        self.register_buffer('class_count',
                             torch.full((num_classes, 1),
                                        cc_init,
                                        dtype=torch.long))
        self.shape = shape

    def hook_fn(self, module, inputs, outputs):
        pass

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.class_count.fill_(1)

    def forward(self, inputs, labels):
        if self.tracking_stats:
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


class CStatsNet(nn.Module):
    def __init__(self, net, class_conditional=True, unbiased_count=True):
        super(CStatsNet, self).__init__()

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
            self.stats.append(CStatistics(n_out,
                                          n_classes,
                                          class_conditional=class_conditional,
                                          unbiased_count=unbiased_count))

        # print(vars(self.stats[0]))
        self.class_conditional = class_conditional
        self.bypass_stats_layer = True

    def forward(self, data):

        use_stats = isinstance(data, dict) and 'labels' in data
        bypass_stats_layer = self.bypass_stats_layer

        if use_stats:
            x = data['inputs']
            labels = data['labels']
        else:
            x = data
            bypass_stats_layer = True  # accidentally didn't switch off

        L = len(self.fc)

        modules = net.modules()  # named modules?

        for i in enumerate(modules):
            # use_stats: False, when ONLY evaluating
            # True, when training and inversion
            # tracking_stats: might be turned off for initial training
            # or after sufficient tracking when done training (1 epoch)
            if not bypass_stats_layer:
                x = self.stats[i](x, labels)
            if i < L:
                x = self.fc[i](x)
            if i < L - 1:
                x = F.relu(x)
        return x

    def __init__(self, net):
        self.net = net
        self.bypass_stats_layer = False

    def stop_tracking_stats(self):
        self.bypass_stats_layer = True
        for m in self.stats:
            m.tracking_stats = False

    def start_tracking_stats(self):
        self.eval()
        self.bypass_stats_layer = False
        for m in self.stats:
            m.tracking_stats = True

    def start_inverting(self):
        self.eval()
        self.bypass_stats_layer = False
        for m in self.stats:
            m.tracking_stats = False

    def collect_stats(self):
        stat_vars = ['running_mean', 'running_var', 'class_count']
        stats = []
        for m in self.stats:
            stat = {}
            for s in stat_vars:
                stat[s] = getattr(m, s).data
            stats.append(stat)
        return stats


def deep_inversion(net, labels, steps=5):

    num_features = 2

    net.start_inverting()

    bs = len(labels)
    inputs = torch.randn((bs, num_features), requires_grad=True)

    optimizer = optim.SGD([inputs], lr=0.1)

    history = []
    history.append(inputs.data.detach().clone())

    for step in range(1, steps + 1):

        net.optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = net(data)
        loss = net.criterion(outputs, labels)

        # reg_loss = sum([s.regularization for s in net.stats])
        reg_loss = net.stats[0].regularization

        loss = loss + reg_loss
        # loss = net.stats[0].regularization

        loss.backward()

        optimizer.step()
        history.append(inputs.data.detach().clone())

    print("Finished Inverting")
    data = torch.stack(history, dim=-1)
    cmap = matplotlib.cm.get_cmap('Spectral')
    labels_max = max(labels).float().item()
    for i, d in enumerate(data):
        plt.plot(d[0], d[1],
                 '--',
                 c=cmap(labels[i].item() / labels_max),
                 linewidth=0.7,
                 alpha=1)

    net.stop_tracking_stats()
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

net = Net01([2, 8, 6, 3], class_conditional=True, unbiased_count=True)
net.compile(lr=0.01)
dataset = Dataset2D(type=0, **dataset_params)
data_loader = dataset.generator(**dataset_gen_params)
# inputs, labels = next(iter(data_loader))
train(net, data_loader, **training_params)
learn_stats(net, data_loader, num_epochs=50)
plot_decision_boundary(dataset, net)
stats = net.collect_stats()
plot_stats_mean(stats[0])
target_labels = torch.arange(3)
invert = deep_inversion(net, target_labels, steps=50)
plot_invert(invert, target_labels)
plt.show()

tb.close()
