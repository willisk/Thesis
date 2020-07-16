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

from utility import plot_decision_boundary
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

            outputs = net(inputs)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            batch_size = len(inputs)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_correct += count_correct(outputs, labels)

        accuracy = total_correct / total_count
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Accuracy', accuracy, epoch)

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, total_loss))
    print("Finished Training")


def learn_stats(stats_net, data_loader, num_epochs=1):

    stats_net.start_tracking_stats()

    batch_total = 1

    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for batch_i, data in enumerate(data_loader, batch_total):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            stats_net(x)

            # for i, layer in enumerate(stats_net.stats, 1):
            #     for j, feature in enumerate(layer.running_mean):
            #         for m, f in enumerate(feature):
            #             tb.add_scalar("stats%i.class%i.f%i" % (i, j, m + 1),
            #                           feature[m], batch_i)

        batch_total = batch_i + 1

    stats_net.disable_hooks()
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
                cmap=cmap, edgecolors='k', alpha=1,
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
            self.fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x):
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


class StatsHook():

    def __init__(self, stats_net, module, num_classes,
                 class_conditional=True, unbiased_count=False):

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = stats_net
        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.unbiased_count = unbiased_count
        self.tracking_stats = True
        self.initialized = False
        self.enabled = False
        self.pre_hook = True  # remove?

    def hook_fn(self, module, inputs, outputs):
        if not self.enabled:
            return
        labels = self.stats_net.current_labels
        if self.pre_hook:
            x = inputs[0]
        else:
            x = outputs
        if self.tracking_stats:
            if not self.initialized:
                num_features = x.shape[1]
                self.init_parameters(num_features)
            cat_cond_mean_(
                x, labels, self.num_classes, self.num_features,
                mean=self.running_mean,
                cc=self.class_count,
                class_conditional=self.class_conditional)
        else:
            if not self.initialized:
                print("Error: Statistics Parameters not initialized")
            means = self.running_mean[labels]
            self.regularization = (x - means).norm(2).sum()

    def init_parameters(self, num_features):
        self.num_features = num_features

        if not self.class_conditional:
            num_classes = 1
        shape = (self.num_classes, self.num_features)

        if self.unbiased_count:
            cc_init = 1
        else:
            cc_init = 0

        self.running_mean = torch.zeros(shape)
        self.running_var = torch.ones(shape)
        self.class_count = torch.full((self.num_classes, 1),
                                      cc_init,
                                      dtype=torch.long)
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True, unbiased_count=True):
        super(CStatsNet, self).__init__()

        self.net = net

        hooks = []
        for i, (name, m) in enumerate(net.named_modules()):
            if isinstance(m, nn.ModuleList) or isinstance(m, nn.CrossEntropyLoss):
                continue
            if i == 0:  # XXX always assume this is neural net??
                continue
            #     pre_hook = True
            # else:
            #     pre_hook = False
            hooks.append(StatsHook(self, m, num_classes,
                                   class_conditional=class_conditional,
                                   unbiased_count=unbiased_count))

        self.hooks = hooks
        self.class_conditional = class_conditional

    def forward(self, data):
        self.current_labels = data['labels']
        return self.net(data['inputs'])

    def stop_tracking_stats(self):
        self.eval()
        for m in self.hooks:
            m.tracking_stats = False

    def start_tracking_stats(self):
        self.enable_hooks()
        self.eval()
        for m in self.hooks:
            m.tracking_stats = True

    def enable_hooks(self):
        for h in self.hooks:
            h.enabled = True

    def disable_hooks(self):
        for h in self.hooks:
            h.enabled = False

    def collect_stats(self):
        stat_vars = ['running_mean', 'running_var', 'class_count']
        stats = []
        for m in self.hooks:
            stat = {}
            for s in stat_vars:
                stat[s] = getattr(m, s).data
            stats.append(stat)
        return stats


def deep_inversion(stats_net, labels, steps=5):

    num_features = 2

    net = stats_net.net
    stats_net.stop_tracking_stats()
    stats_net.enable_hooks()

    bs = len(labels)
    inputs = torch.randn((bs, num_features), requires_grad=True)

    optimizer = optim.SGD([inputs], lr=0.1)

    history = []
    history.append(inputs.data.detach().clone())

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = stats_net(data)
        loss = net.criterion(outputs, labels)

        # reg_loss = sum([s.regularization for s in net.stats])
        # reg_loss = stats_net.hooks[0].regularization

        # loss = loss + reg_loss
        loss = stats_net.hooks[0].regularization

        loss.backward()

        optimizer.step()
        history.append(inputs.data.detach().clone())

    print("Finished Inverting")
    data = torch.stack(history, dim=-1)
    cmap = matplotlib.cm.get_cmap('Spectral')
    labels_max = max(labels).float().item()
    for i, d in enumerate(data):
        c = cmap(labels[i].item() / labels_max)
        plt.plot(d[0], d[1], '--', c=c, linewidth=0.7, alpha=1)
        # plt.plot(d[0], d[1], 'x', c=c, alpha=0.3)

    stats_net.disable_hooks()
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


dataset = Dataset2D(type=0, **dataset_params)
data_loader = dataset.generator(**dataset_gen_params)
num_classes = dataset.get_num_classes()

net = Net01([2, 8, 6, num_classes])
net.compile(lr=0.01)

train(net, data_loader, **training_params)

stats_net = CStatsNet(net, num_classes, class_conditional=True)
learn_stats(stats_net, data_loader, num_epochs=50)

plot_decision_boundary(dataset, net)
stats = stats_net.collect_stats()
plot_stats_mean(stats[0])

target_labels = torch.arange(num_classes)
invert = deep_inversion(stats_net, target_labels, steps=50)
plot_invert(invert, target_labels)
plt.show()

tb.close()
