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

sys.path.append(PWD)

import utility
import statsnet
import nets
import deepinversion
importlib.reload(utility)
importlib.reload(statsnet)
importlib.reload(nets)
importlib.reload(deepinversion)

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

    def train_loader(self):
        return torch.utils.data.DataLoader(self, **self.dataset_gen_params)

    def test_loader(self):
        return torch.utils.data.DataLoader(self.test_set, **self.dataset_gen_params)

    def full(self):
        return self.X, self.Y

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def pretrained_statsnet(self, net, name, hook_before_bn=False, resume_training=False, use_drive=False, input_shape=None):

        criterion = self.get_criterion()
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        num_classes = self.get_num_classes()

        stats_net = statsnet.CStatsNet(
            net, num_classes, hook_before_bn=hook_before_bn, class_conditional=True)

        if input_shape is not None:
            init_sample = torch.zeros(input_shape)
        else:
            init_sample = next(iter(self.train_loader()))[0]
        stats_net.init_hooks(init_sample)

        net_path = os.path.join(MODELDIR, "net_" + name + ".pt")
        csnet_path = os.path.join(MODELDIR, "csnet_" + name + ".pt")

        csnet_save_path, csnet_load_path = csnet_path, csnet_path

        if use_drive:
            csnet_save_path, csnet_load_path = utility.search_drive(csnet_path)

        pretrained = False
        if os.path.exists(csnet_load_path):
            checkpoint = torch.load(csnet_load_path)
            stats_net.load_state_dict(checkpoint, strict=True)
            print("CSNet loaded: " + csnet_load_path)
            pretrained = True

        if resume_training or not pretrained:
            data_loader = self.train_loader()
            stats_net.disable_hooks()
            utility.train(net, data_loader, criterion, optimizer,
                          model_path=net_path, resume_training=resume_training,
                          use_drive=use_drive,
                          ** self.training_params)
            stats_net.enable_hooks()

            if not pretrained:
                utility.learn_stats(stats_net, data_loader)
            torch.save(stats_net.state_dict(), csnet_save_path)
            print("CSNet saved: " + csnet_save_path)

        stats_net.stop_tracking_stats()

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

    def load_statsnet(self, resume_training=False, use_drive=False):
        layer_dims = [4, 16, 16, self.get_num_classes()]
        net = nets.FCNet(layer_dims)
        stats_net = self.pretrained_statsnet(
            net, "iris", resume_training=resume_training, use_drive=use_drive)
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

    def load_statsnet(self, resume_training=False, use_drive=False):
        layer_dims = [16, 32, 32, 16, 8 * 8 * 16, self.get_num_classes()]
        net = nets.ConvNet(layer_dims, 3)
        stats_net = self.pretrained_statsnet(
            net, "digits", resume_training=resume_training, use_drive=use_drive)
        return stats_net

    def plot(self, net):
        idx = np.arange(9)
        X = self.X[idx]
        Y = self.Y[idx]
        pred = net.predict(X).numpy()
        colors = np.array(['red', 'green'])[(pred == Y).astype(int)]
        utility.make_grid(X, pred, "pred: {}", colors=colors)
        plt.show()

    def plot_history(self, invert, labels):
        utility.make_grid(invert, labels, "target: {}")


import torchvision
import torchvision.transforms as transforms


class DatasetCifar10(torchvision.datasets.CIFAR10, Dataset):
    def __init__(self, load_dataset=True):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if load_dataset:
            super().__init__(root=DATADIR, train=True,
                             download=True, transform=transform)
            self.test_set = torchvision.datasets.CIFAR10(root=DATADIR, train=True,
                                                         download=True, transform=transform)

        Dataset.__init__(self)

        self.classes = np.array(['plane', 'car', 'bird', 'cat',
                                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

        self.training_params['num_epochs'] = 50
        self.training_params['print_every'] = 1
        self.training_params['save_every'] = 1

    def get_num_classes(self):
        return 10

    def load_statsnet(self, net=None, name="cifar10-resnet", resume_training=False, use_drive=False):
        if net is None:
            net = nets.ResNet20(10)
            name = "cifar10-resnet20"
        input_shape = [64, 3, 32, 32]
        stats_net = self.pretrained_statsnet(
            net, name, input_shape=input_shape, hook_before_bn=True, resume_training=resume_training, use_drive=use_drive)
        return stats_net

    def print_accuracy(self, net):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader():
                images, labels = data
                pred = net.predict(images)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        acc = correct / total
        print("Accuracy on test set: {:3f}".format(acc))

    def to_image_plt(self, im):
        return im.permute(0, 2, 3, 1) / 2 + 0.5

    def plot(self, net):
        print("random samples:")
        data, labels = next(iter(self.test_loader()))
        idx = np.arange(9)
        X = data[idx]
        Y = labels[idx].numpy()
        pred = net.predict(X).numpy()
        colors = np.array(['red', 'green'])[(pred == Y).astype(int)]

        utility.make_grid(self.to_image_plt(X), self.classes[pred],
                          "pred: {}", colors=colors)
        plt.show()

    def plot_stats(self, stats_net):
        stats = stats_net.collect_stats()[0]
        mean = stats['running_mean']
        var = stats['running_var']

        num_classes = self.get_num_classes()
        print("dataset means:")
        utility.make_grid(self.to_image_plt(mean),
                          self.classes, "{}")

    def plot_history(self, invert, labels):
        frames = []
        for X, step in invert:
            frame = utility.make_grid(self.to_image_plt(X),
                                      description="step={}".format(step),
                                      labels=self.classes[labels],
                                      title_fmt="target: {}")
            frames.append(frame)
        return frames


class Dataset2D(Dataset):

    def __init__(self, type):
        params = {
            'n_samples': 500,
            # 'noise': 0.1,
            # 'factor': 0.5
        }
        super().__init__()

        self.training_params['num_epochs'] = 500
        self.training_params['print_every'] = 100
        # self.training_params['save_every'] = 1

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

    def load_statsnet(self, resume_training=False, use_drive=False):
        layer_dims = [2, 8, 6, self.get_num_classes()]
        net = nets.FCNet(layer_dims)
        name = "toy_{}".format(self.type)
        stats_net = self.pretrained_statsnet(
            net, name, resume_training=resume_training, use_drive=use_drive)
        return stats_net

    # def plot_uq(self, stats_net, loss_fn, cmap='bone'):

    #     X, Y = self.X, self.Y
    #     with torch.no_grad():
    #         utility.plot_contourf_data(
    #             X, loss_fn, n_grid=100, scale_grid=1.5, cmap=cmap, levels=30,
    #             contour=True, colorbar=True)
    #     plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap='Spectral', alpha=.4)

    def plot(self, net=None):
        X, Y = self.X, self.Y
        if net is not None:
            with torch.no_grad():
                utility.plot_contourf_data(X, net.predict, contour=True)
        plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap='Spectral', alpha=.4)

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
            plt.scatter(m[0], m[1], color=c,
                        edgecolors='k', alpha=0.5,
                        marker='^')

    def plot_history(self, history, labels):
        data_list = [d for d, i in history]
        data = torch.stack(data_list, dim=-1)
        cmap = matplotlib.cm.get_cmap('Spectral')
        labels_max = max(labels).float().item()
        for i, d in enumerate(data):
            c = cmap(labels[i].item() / labels_max)
            plt.plot(d[0], d[1], '--', c=c, linewidth=0.7, alpha=1)
            plt.plot(d[0], d[1], '.', c='k', markersize=2, alpha=1)
        x = data_list[-1].detach().T
        plt.scatter(x[0], x[1], c='k', alpha=1, marker='x')
        plt.scatter(x[0], x[1],
                    c=labels, cmap=cmap, alpha=0.9, marker='x')
