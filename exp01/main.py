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
from sklearn.datasets import make_blobs, make_circles, make_moons

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utility import plot_decision_boundary
# from datasets import Dataset2D

np.random.seed(0)
torch.manual_seed(0)

comment = ""
tb = SummaryWriter(comment=comment)

# %% DATASET


class Dataset2D(torch.utils.data.Dataset):

    def __init__(self, dataset_type, **params):
        if dataset_type == 0:
            X, Y = make_blobs(n_features=2, centers=3, **params)
            for i, _ in enumerate(X):
                if Y[i] == 0:
                    X[i] = X[i] * 2
        if dataset_type == 1:
            X, Y = make_circles(noise=.1, **params)
            for i, _ in enumerate(X):
                if Y[i] == 0:
                    X[i][0] = X[i][0] * 2
            X = X * 3
        if dataset_type == 2:
            X, Y = make_moons(noise=.2, **params)
        if dataset_type == 3:
            X, Y = make_blobs(n_features=2, centers=25,
                              cluster_std=0.2, **params)
            Y = Y % 2

        X = np.array(X, dtype='float32')
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def get_num_classes(self):
        return max(self.Y).item() + 1

    def mean(self):
        return [self.X[self.Y == y].mean(axis=0)
                for y in range(self.get_num_classes())]

    def var(self):
        return [self.X[self.Y == y].var(axis=0)
                for y in range(self.get_num_classes())]

    def generator(self, **params):
        return torch.utils.data.DataLoader(self, **params)

    def full(self):
        return self.X, self.Y

# %% utility


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum()


def plot_decision_boundary(dataset, net, contourgrad=False):

    cmap = 'Spectral'
    X, Y = dataset.full()
    h = 0.05
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = (np.c_[xx.ravel(), yy.ravel()])
    mesh = torch.from_numpy(mesh.astype('float32'))
    Z = net.predict(mesh)
    Z = Z.T.reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    if contourgrad:
        A = net(mesh)
        plt.contourf(xx, yy, A.T.reshape(xx.shape), cmap=cmap, alpha=.3)
    else:
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=.3)
    plt.contour(xx, yy, Z, colors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap=cmap, alpha=.4)


def plot_stats_mean(stats):
    cmap = matplotlib.cm.get_cmap('Spectral')
    mean = stats['running_mean']
    var = stats['running_var']

    size = torch.sqrt(var) * 2

    num_classes = mean.shape[0]
    # mean = mean.T
    if num_classes == 1:    # no class_conditional
        c = 'k'
    else:
        colors = cmap(np.arange(num_classes) / (num_classes - 1.))
    Ellipse = matplotlib.patches.Ellipse
    for m, v, c in zip(mean, size, colors):
        ell = Ellipse(m, v[0], v[1], edgecolor=c, lw=1, fill=False)
        plt.gca().add_artist(ell)
        # plt.scatter(mean[0], mean[1], c=c,
        #         cmap=cmap, edgecolors='k', alpha=0.5,
        #         s=var,
        #         marker='^')


def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, var, cc,
                   class_conditional=True, bessel_correction=True):

    if class_conditional:
        total = torch.zeros((n_classes, n_features))
        total.index_add_(0, labels, inputs)
        N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)
    else:
        total = inputs.sum(dim=0)
        N_class = len(inputs)

    cc_f = cc.float()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)
    ##
    total_2 = torch.zeros((n_classes, n_features))
    # total_var.index_add_(0, labels, inputs - mean[labels])
    for i, x in enumerate(inputs):
        total_2[labels[i]] += (inputs[i])**2
    var.mul_(cc_f / cc)
    var.add_((total_2 - N_class * mean**2) / cc)


def train(net, data_loader, criterion, optimizer, num_epochs=1, print_every=10):
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

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_correct += count_correct(outputs, labels)

        accuracy = total_correct / total_count
        # tb.add_scalar('Loss', total_loss, epoch)
        # tb.add_scalar('Accuracy', accuracy, epoch)

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

    def predict(self, x):
        net.eval()
        y = self.__call__(x)
        return torch.argmax(y, dim=-1)


class StatsHook(nn.Module):

    def __init__(self, stats_net, module, num_classes,
                 class_conditional=True, bessel_correction=False):
        super(StatsHook, self).__init__()

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = [stats_net]
        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.bessel_correction = bessel_correction
        self.tracking_stats = True
        self.initialized = False
        self.enabled = False

    def hook_fn(self, module, inputs, outputs):
        if not self.enabled:
            return
        labels = self.stats_net[0].current_labels
        x = inputs[0]
        if self.tracking_stats:
            if not self.initialized:
                num_features = x.shape[1]
                self.init_parameters(num_features)
            cat_cond_mean_(x.detach(), labels, self.num_classes, self.num_features,
                           mean=self.running_mean,
                           var=self.running_var,
                           cc=self.class_count,
                           class_conditional=self.class_conditional,
                           bessel_correction=self.bessel_correction)
        else:   # inverting
            if not self.initialized:
                print("Error: Statistics Parameters not initialized")
            means = self.running_mean[labels]
            self.regularization = (x - means).norm(2).sum()

    def init_parameters(self, num_features):
        self.num_features = num_features

        if not self.class_conditional:
            num_classes = 1
        else:
            num_classes = self.num_classes
        shape = (num_classes, self.num_features)

        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.register_buffer('class_count', torch.zeros((num_classes, 1),
                                                        dtype=torch.long))
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True, bessel_correction=True):
        super(CStatsNet, self).__init__()

        self.net = net

        self.hooks = nn.ModuleList()
        for i, (name, m) in enumerate(net.named_modules()):
            if isinstance(m, nn.ModuleList) or isinstance(m, nn.CrossEntropyLoss):
                continue
            if i == 0:  # XXX always assume this is neural net??
                continue
            self.hooks.append(StatsHook(self, m, num_classes,
                                        class_conditional=class_conditional,
                                        bessel_correction=bessel_correction))

        # self.class_conditional = class_conditional

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

    optimizer = optim.Adam([inputs], lr=0.1)

    history = []
    history.append(inputs.data.detach().clone())

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = stats_net(data)
        loss = criterion(outputs, labels)

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
    x = inputs.detach().T
    plt.scatter(x[0], x[1], c='k', alpha=1, marker='x')
    plt.scatter(x[0], x[1],
                c=labels, cmap=cmap, alpha=0.9, marker='x')

    stats_net.disable_hooks()
    return inputs


training_params = {'num_epochs': 200,
                   'print_every': 20}
dataset_params = {
    'dataset_type': 3,
    'n_samples': 500,
    # 'noise': 0.1,
    # 'factor': 0.5
}
dataset_gen_params = {'batch_size': 64,
                      'shuffle': True}


dataset = Dataset2D(**dataset_params)
data_loader = dataset.generator(**dataset_gen_params)
num_classes = dataset.get_num_classes()

net = Net01([2, 8, 6, num_classes])
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train(net, data_loader, criterion, optimizer, **training_params)

stats_net = CStatsNet(net, num_classes, class_conditional=True)
learn_stats(stats_net, data_loader, num_epochs=50)


plot_decision_boundary(dataset, net)
plot_stats_mean(stats_net.collect_stats()[0])

target_labels = torch.arange(num_classes) % num_classes
invert = deep_inversion(stats_net, target_labels, steps=100)
# plot_invert(invert, target_labels)
# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
plt.show()

tb.close()
