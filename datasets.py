import os
import sys
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility
import statsnet
importlib.reload(utility)
importlib.reload(statsnet)

training_params = {'num_epochs': 200,
                   'print_every': 20}
dataset_params = {
    'n_samples': 500,
    # 'noise': 0.1,
    # 'factor': 0.5
}
dataset_gen_params = {'batch_size': 64,
                      'shuffle': True}


def load2D(type=1):

    dataset = Dataset2D(type, **dataset_params)
    num_classes = dataset.get_num_classes()

    net = Net01([2, 8, 6, num_classes])
    path = "csnet{}.pt".format(type)

    stats_net = dataset.pretrained_statsnet(net, path)

    return stats_net, dataset


def loadIris():
    dataset = DatasetIris()


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

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

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def pretrained_statsnet(self, net, path):
        data_loader = self.generator(**dataset_gen_params)

        criterion = self.get_criterion()
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        num_classes = self.get_num_classes()

        stats_net = statsnet.CStatsNet(
            net, num_classes, class_conditional=True)
        stats_net.init_hooks(data_loader)

        if os.path.exists(path):
            checkpoint = torch.load(path)
            stats_net.load_state_dict(checkpoint['csnet_state_dict'])
            print("Model loaded: {}".format(path))
        else:
            utility.train(net, data_loader, criterion,
                          optimizer, **training_params)
            utility.learn_stats(stats_net, data_loader, num_epochs=50)
            torch.save({
                'epoch': training_params['num_epochs'],
                'csnet_state_dict': stats_net.state_dict(),
            }, path)
            print("Model saved: {}".format(path))
        return stats_net


class DatasetIris(Dataset):
    def __init__(self, type, **params):
        iris_data = load_iris()
        self.X, self.Y = iris_data.data, iris_data.target


class Dataset2D(Dataset):

    def __init__(self, type, **params):
        if type == 0:
            X, Y = make_blobs(n_features=2, centers=3, **params)
            for i, _ in enumerate(X):
                if Y[i] == 0:
                    X[i] = X[i] * 2
        if type == 1:
            X, Y = make_circles(noise=.1, **params)
            for i, _ in enumerate(X):
                if Y[i] == 0:
                    X[i][0] = X[i][0] * 2
            X = X * 3
        if type == 2:
            X, Y = make_moons(noise=.2, **params)
        if type == 3:
            X, Y = make_blobs(n_features=2, centers=25,
                              cluster_std=0.2, **params)
            Y = Y % 2

        X = np.array(X, dtype='float32')
        self.X, self.Y = X, Y

    def plot(self, net, contourgrad=False):

        cmap = 'Spectral'
        X, Y = self.full()
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

    def plot_stats(self, stats_net):
        cmap = matplotlib.cm.get_cmap('Spectral')
        stats = stats_net.collect_stats()[0]
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

    def plot_history(self, history, labels):
        data = torch.stack(history, dim=-1)
        cmap = matplotlib.cm.get_cmap('Spectral')
        labels_max = max(labels).float().item()
        for i, d in enumerate(data):
            c = cmap(labels[i].item() / labels_max)
            plt.plot(d[0], d[1], '--', c=c, linewidth=0.7, alpha=1)
            # plt.plot(d[0], d[1], 'x', c=c, alpha=0.3)
        x = history[-1].detach().T
        plt.scatter(x[0], x[1], c='k', alpha=1, marker='x')
        plt.scatter(x[0], x[1],
                    c=labels, cmap=cmap, alpha=0.9, marker='x')


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
        self.eval()
        y = self.__call__(x)
        return torch.argmax(y, dim=-1)
