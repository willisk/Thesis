import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, load_digits


import torch
import torch.nn as nn
import torch.optim as optim

import importlib

PWD = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(PWD, "models")
DATADIR = os.path.join(PWD, "data")

if not os.path.exists(MODELDIR):
    os.mkdir(MODELDIR)

sys.path.append(PWD)

import utility
import statsnet
import nets
importlib.reload(utility)
importlib.reload(statsnet)
importlib.reload(nets)

training_params = {'num_epochs': 200,
                   'print_every': 20}
dataset_gen_params = {'batch_size': 64,
                      'shuffle': True}


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.training_params = training_params
        self.dataset_gen_params = dataset_gen_params

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

    def print_accuracy(self, net):
        inputs, labels = torch.Tensor(self.X), torch.Tensor(self.Y)
        preds = net(inputs).argmax(dim=-1)
        acc = (preds == labels).type(torch.FloatTensor).mean()
        print("Accuracy: {:3f}".format(acc))

    def train_loader(self, **params):
        return torch.utils.data.DataLoader(self, **params)

    def full(self):
        return self.X, self.Y

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def pretrained_statsnet(self, net, name):
        data_loader = self.train_loader(**self.dataset_gen_params)

        criterion = self.get_criterion()
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        num_classes = self.get_num_classes()

        stats_net = statsnet.CStatsNet(
            net, num_classes, class_conditional=True)
        stats_net.init_hooks(data_loader)

        net_path = os.path.join(MODELDIR, "net_" + name + ".pt")
        csnet_path = os.path.join(MODELDIR, "csnet_" + name + ".pt")

        if os.path.exists(csnet_path):
            checkpoint = torch.load(csnet_path)
            stats_net.load_state_dict(checkpoint)
            print("CSNet loaded: " + csnet_path)
        else:
            if os.path.exists(net_path):
                checkpoint = torch.load(net_path)
                net.load_state_dict(checkpoint['net_state_dict'])
            else:
                print("Beginning training.")
                utility.train(net, data_loader, criterion, optimizer,
                              model_path=net_path,
                              **self.training_params)
                print("Beginning tracking stats.")
                utility.learn_stats(stats_net, data_loader, num_epochs=2)
                torch.save(stats_net.state_dict(), csnet_path)
                print("CSNet saved: {}".format(csnet_path))
        return stats_net

    def load_statsnet(self):
        pass

    def plot(self, net):
        pass

    def plot_stats(self, stats_net):
        pass

    def plot_history(self, history, labels):
        pass


class DatasetIris(Dataset):
    def __init__(self):
        super().__init__()
        self.iris = load_iris()
        self.X = np.array(self.iris['data'], dtype='float32')
        self.Y = self.iris['target']

    def load_statsnet(self):
        layer_dims = [4, 16, 16, self.get_num_classes()]
        net = nets.FCNet(layer_dims)
        stats_net = self.pretrained_statsnet(net, "iris")
        return stats_net

    def plot(self, net):
        self.fig = utility.scatter_matrix(self.X, self.Y,
                                          feature_names=self.iris['feature_names'],
                                          s=10,
                                          alpha=0.7,
                                          cmap='brg')

    def plot_history(self, invert, labels):
        utility.scatter_matrix(invert, labels,
                               fig=self.fig,
                               marker='x',
                               c='k',
                               s=60)
        utility.scatter_matrix(invert, labels,
                               fig=self.fig,
                               marker='x',
                               s=50,
                               cmap='brg')


class DatasetDigits(Dataset):
    def __init__(self):
        super().__init__()
        digits = load_digits()
        self.X = torch.FloatTensor(digits['data']).reshape(-1, 1, 8, 8)
        self.Y = digits['target']

        self.training_params['save_every'] = 20

    def load_statsnet(self):
        layer_dims = [16, 32, 32, 16, 8 * 8 * 16, self.get_num_classes()]
        net = nets.ConvNet(layer_dims, 3)
        stats_net = self.pretrained_statsnet(net, "digits")
        return stats_net

    def plot(self, net):
        idx = np.arange(9)
        X = self.X[idx]
        Y = self.Y[idx]
        pred = net.predict(X).numpy()
        colors = np.array(['red', 'green'])[(pred == Y).astype(int)]
        utility.plot_num_matrix(X, pred, "pred: {}", colors)
        plt.show()

    def plot_history(self, invert, labels):
        utility.plot_num_matrix(invert, labels, "target: {}")


import torchvision
import torchvision.transforms as transforms


class DatasetCifar10(torchvision.datasets.CIFAR10, Dataset):
    def __init__(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        super().__init__(root='../data', train=True,
                         download=True, transform=transform)

        Dataset.__init__(self)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.training_params['num_epochs'] = 10
        self.training_params['print_every'] = 1
        self.training_params['save_every'] = 1

    def train_loader(self, **params):
        return torch.utils.data.DataLoader(self, **params)

    def get_num_classes(self):
        return 10

    def load_statsnet(self):
        net = nets.ResNet20(10)
        stats_net = self.pretrained_statsnet(net, "cifar10-resnet20")
        return stats_net


class DatasetImagenet(Dataset):

    def __init__(self):
        self.imagenet_data = torchvision.datasets.ImageNet(
            'imagenet_data', split='train', download=True)
        digits = load_digits()
        self.X = torch.FloatTensor(digits['data']).reshape(-1, 1, 8, 8)
        self.Y = digits['target']

    def train_loader(self, **params):
        return torch.utils.data.DataLoader(self.imagenet_data, **params)

    def load_statsnet(self):
        # net = torchvision.models.resnet50(pretrained=True)
        # print("pretrained")
        net = nets.ResNet20(10)
        stats_net = self.pretrained_statsnet(net, "Imagenet-resnet20")
        return stats_net

    def plot(self, net):
        idx = np.arange(9)
        X = self.X[idx]
        Y = self.Y[idx]
        pred = net.predict(X).numpy()
        colors = np.array(['red', 'green'])[(pred == Y).astype(int)]
        utility.plot_num_matrix(X, pred, "pred: {}", colors)
        plt.show()

    def plot_history(self, invert, labels):
        utility.plot_num_matrix(invert, labels, "target: {}")


class Dataset2D(Dataset):

    def __init__(self, type):
        params = {
            'n_samples': 500,
            # 'noise': 0.1,
            # 'factor': 0.5
        }
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
        self.type = type
        self.X, self.Y = X, Y

    def load_statsnet(self):
        layer_dims = [2, 8, 6, self.get_num_classes()]
        net = nets.FCNet(layer_dims)
        filename = "csnet{}.pt".format(self.type)
        stats_net = self.pretrained_statsnet(net, filename)
        return stats_net

    def plot(self, net, contourgrad=False):
        utility.plot_prediction2d(self.X, self.Y, net)

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
