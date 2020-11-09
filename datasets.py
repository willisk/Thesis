import os
import sys

import numpy as np
from numpy.linalg import det, inv

import matplotlib
import matplotlib.pyplot as plt

from scipy.special import logsumexp
from scipy.stats import multivariate_normal, ortho_group
from sklearn.datasets import make_blobs, make_circles, make_moons, load_iris, load_digits

import torch
import torch.nn as nn
import torch.optim as optim

PWD = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(PWD, "models")
DATADIR = os.path.join(PWD, "data")

sys.path.append(PWD)

import nets
import utility
import statsnet
import deepinversion
if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(nets)
    importlib.reload(utility)
    importlib.reload(statsnet)
    importlib.reload(deepinversion)

training_params = {'num_epochs': 200}
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
        if not hasattr(self, "num_classes"):
            self.num_classes = max(self.Y).item() + 1
        return self.num_classes

    def print_accuracy(self, net):
        utility.print_net_accuracy(net,
                                   torch.Tensor(self.X), torch.Tensor(self.Y))

    def train_loader(self):
        return torch.utils.data.DataLoader(self, **self.dataset_gen_params)

    def test_loader(self):
        return torch.utils.data.DataLoader(self.test_set, **self.dataset_gen_params)

    def full(self):
        return self.X, self.Y

    def get_criterion(self, reduction='mean'):
        return nn.CrossEntropyLoss(reduction=reduction)

    def pretrained_statsnet(self, net, name, resume_training=False, use_drive=False, input_shape=None):

        # XXX lr should be in training params
        criterion = self.get_criterion()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        num_classes = self.get_num_classes()

        stats_net = statsnet.CStatsNet(net, num_classes)

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
        stats_net.verify_hooks_finite()

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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if load_dataset:
            super().__init__(root=DATADIR, train=True, download=True, transform=self.transform)
            self.test_set = torchvision.datasets.CIFAR10(root=DATADIR, train=False,
                                                         download=True, transform=self.transform)

        Dataset.__init__(self)

        self.classes = np.array(['plane', 'car', 'bird', 'cat',
                                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

        self.training_params['num_epochs'] = 50
        self.training_params['save_every'] = 1

    def full(self):
        X = Y = None
        for inputs, labels in self.train_loader():
            if X is None:
                X = inputs.clone()
                Y = labels
            else:
                X = torch.cat((X, inputs))
                Y = torch.cat((Y, labels))
        return X, Y

    def get_num_classes(self):
        return 10

    def load_statsnet(self, net=None, name="cifar10-resnet", resume_training=False, use_drive=False):
        if net is None:
            net = nets.ResNet20(10)
            name = "cifar10-resnet20"
        input_shape = [1, 3, 32, 32]
        stats_net = self.pretrained_statsnet(
            net, name, input_shape=input_shape, resume_training=resume_training, use_drive=use_drive)
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
            frame = utility.make_grid(self.to_image_plt(X).cpu(),
                                      #   labels=self.classes[labels],
                                      description="step={}".format(step),
                                      #   title_fmt="target: {}"
                                      )
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

    def plot(self, net=None, data=None, loss_fn=None, cmap='Spectral', scatter=True, legend=False):
        if data is None:
            X, Y = self.X, self.Y
        else:
            X, Y = data['inputs'], data['labels']
        assert X.shape[1] == 2, "Cannot plot n-dimensional."
        scale_grid, levels = 1.5, 30
        if loss_fn is None and net is not None:
            loss_fn, scale_grid, levels = net.predict, 1, None
        if loss_fn is not None:
            with torch.no_grad():
                utility.plot_contourf_data(
                    X, loss_fn, n_grid=200, scale_grid=scale_grid,
                    cmap=cmap, levels=levels, contour=True, colorbar=True)
        if scatter:
            scatter = plt.scatter(X[:, 0], X[:, 1],
                                  c=Y.squeeze(), cmap='Spectral', alpha=.4)
            if legend:
                leg_obj = plt.legend(
                    *scatter.legend_elements(), title="Classes")
                plt.gca().add_artist(leg_obj)

    def plot_stats(self, stats_net=None):
        stats = stats_net.collect_stats()[0]
        mean = stats['running_mean']
        var = stats['running_var']
        utility.plot_stats(mean, var)

    def plot_history(self, history, labels):
        data_list = [d for d, i in history]
        data = torch.stack(data_list, dim=-1)
        cmaps = utility.categorical_colors(self.get_num_classes())
        labels_max = max(labels).float().item()
        for i, d in enumerate(data):
            c = cmaps[int(labels[i].item())]
            plt.plot(d[0], d[1], '--', c=c, linewidth=0.7, alpha=1)
            plt.plot(d[0], d[1], '.', c='k', markersize=2, alpha=1)
        x = data_list[-1].detach().T
        plt.scatter(x[0], x[1], c='k', alpha=1, marker='x')
        plt.scatter(x[0], x[1],
                    c=labels, cmap=cmaps, alpha=0.9, marker='x')


class DatasetGMM(Dataset2D):

    def __init__(self, n_dims=30, n_classes=10, n_modes=8, scale_mean=1, scale_cov=1, mean_shift=0, n_samples_per_class=10000, weights=None):
        # pylint: disable=bad-super-call
        super(Dataset2D, self).__init__()

        self.training_params['num_epochs'] = 500

        # # using equal weights for now
        # if weights is None:
        #     weights = torch.ones((n_classes))
        # self.weights = weights / weights.sum()

        self.gmms = [
            random_gmm(n_modes, n_dims, scale_mean, scale_cov, mean_shift)
            for _ in range(n_classes)
        ]

        self.X, self.Y = self.sample(n_samples_per_class)

    def sample(self, n_samples_per_class):
        X = Y = None
        for c, gmm in enumerate(self.gmms):
            x = gmm.sample(n_samples_per_class)
            y = torch.LongTensor([c] * n_samples_per_class)
            X = torch.cat((X, x), dim=0) if X is not None else x
            Y = torch.cat((Y, y), dim=0) if Y is not None else y
        return X, Y

    def log_likelihood(self, X, Y):
        n_total = len(Y) + 0.
        log_lh = 0.
        for c, gmm in enumerate(self.gmms):
            c_mask = Y == c
            log_lh += gmm.logpdf(X[c_mask],
                                 weight_factor=self.weights[c]).sum()
        return log_lh / n_total

    def pairwise_cross_entropy(self):
        if len(self.gmms) < 2:
            return
        print("0 < H < inf")
        for i, P in enumerate(self.gmms):
            Q_rest = combine_gmms([Q for Q in self.gmms if Q != P])
            print(f"H(P{i}, Q_rest) = {P.cross_entropy_sample(Q_rest)}")

    def pairwise_JS(self):
        if len(self.gmms) < 2:
            return
        print("0 < JS < ln(2)")
        M = combine_gmms(self.gmms)
        for i, P in enumerate(self.gmms):
            Q_rest = combine_gmms([Q for Q in self.gmms if Q != P])
            print(f"JS(P{i}|Q_rest) = {P.JS_sample(Q_rest, M=M)}")

    def pdf(self, X, Y=None):
        return torch.exp(self.log_pdf(X, Y))

    def log_pdf(self, X, Y=None):
        X_c = torch.as_tensor(X)
        if Y is not None:
            Y = torch.as_tensor(Y)
        n_class = len(self.gmms)
        a, b = [None] * n_class, [None] * n_class
        a_max = -float('inf')
        for c, gmm, in enumerate(self.gmms):
            if Y is not None:
                X_c = X[Y == c]
                if len(X_c) == 0:
                    continue
                p_c = 1
            else:
                # estimated class prob, should use equal
                p_c = (self.Y == c).sum().item() / len(self.Y)
            n_modes, n_c = len(gmm.means), len(X_c)
            b[c] = gmm.weights.reshape(n_modes, -1) * p_c
            a[c] = torch.empty((n_modes, n_c))
            for m in range(n_modes):
                a[c][m] = torch.as_tensor(gmm.mvns[m].logpdf(X_c))
            a_max = max(a_max, torch.max(a[c]).item())
        sum_exp = torch.zeros((len(X)))
        for c, (log_p_c_X, w) in enumerate(zip(a, b)):
            if log_p_c_X is None:
                continue
            s = (w * torch.exp(log_p_c_X - a_max)).sum(dim=0)
            if Y is not None:
                sum_exp[Y == c] = s
            else:
                sum_exp = sum_exp + s
        return torch.log(sum_exp) + a_max

    def cross_entropy(self, X, Y):
        return -self.log_pdf(X, Y).mean()

    # def concatenate(self, except_for=None):
    #     gmms = [g for g in self.gmms if g is not except_for]
    #     means, covs, weights = zip(
    #         *[(q.means, q.covs, q.weights) for q in gmms])
    #     means = torch.cat(means, dim=0)
    #     covs = torch.cat(covs, dim=0)
    #     weights = torch.cat(weights, dim=0)
    #     weights /= weights.sum()
    #     return GMM(means, covs, weights)

    def load_statsnet(self, resume_training=False, use_drive=False):
        n_dims = self.X.shape[1]
        layer_dims = [n_dims, 32, 32, 32, self.get_num_classes()]
        net = nets.FCNet(layer_dims)
        name = f"GMM_{n_dims}"
        stats_net = self.pretrained_statsnet(
            net, name, resume_training=resume_training, use_drive=use_drive)
        return stats_net


def make_spd_matrix(n_dims):
    D = torch.diag(torch.abs(torch.rand((n_dims))))
    if n_dims == 1:
        return D
    Q = torch.as_tensor(ortho_group.rvs(n_dims), dtype=D.dtype)
    return Q.T @ D @ Q


def random_gmm(n_modes, n_dims, scale_mean=1, scale_cov=1, mean_shift=0):
    weights = torch.rand((n_modes))
    shift = torch.randn((n_dims)) * mean_shift
    means = torch.randn((n_modes, n_dims)) * scale_mean + shift
    covs = torch.empty((n_modes, n_dims, n_dims))
    for i in range(n_modes):
        covs[i] = make_spd_matrix(n_dims) * scale_cov
    return GMM(means, covs, weights)


def combine_gmms(Q_list, p=None):
    means, covs, weights = zip(*[(Q.means, Q.covs, Q.weights) for Q in Q_list])
    means = torch.cat(means, dim=0)
    covs = torch.cat(covs, dim=0)
    if p is None:
        weights = torch.cat(weights, dim=0)
    else:
        weights = torch.cat([p * w for w, p in zip(weights, p)], dim=0)
    return GMM(means, covs, weights)


class GMM():

    def __init__(self, means, covs, weights=None):
        self.means = means
        self.covs = covs
        n_modes = len(means)
        if weights is None:
            weights = torch.ones((n_modes))
        self.weights = weights / weights.sum()
        self.mvns = []
        for i in range(n_modes):
            mvn = multivariate_normal(
                mean=means[i].numpy(), cov=covs[i].numpy())
            self.mvns.append(mvn)

    def __add__(self, Q):
        means = torch.cat((self.means, Q.means), dim=0)
        covs = torch.cat((self.covs, Q.covs), dim=0)
        weights = torch.cat((self.weights, Q.weights), dim=0)
        weights /= 2
        return GMM(means, covs, weights)

    def sample(self, n_samples):
        n_modes, n_dims = self.means.shape
        n_samples = int(n_samples)
        weights = self.weights.to(torch.double)
        choice_mode = torch.as_tensor(np.random.choice(
            n_modes, size=(n_samples), p=weights / weights.sum()))
        samples = torch.empty((n_samples, n_dims))
        for mode in range(n_modes):
            mask = choice_mode == mode
            n_mask = mask.sum().item()
            if n_mask != 0:
                samples[mask] = torch.as_tensor(self.mvns[mode].rvs(
                    size=n_mask).reshape(-1, n_dims), dtype=torch.float)
        return samples

    def log_pdf(self, X, prob_factor=1):
        n_modes, n_samples = len(self.means), len(X)
        log_p_m_X = torch.empty((n_modes, n_samples))
        for i in range(n_modes):
            log_p_m_X[i] = torch.as_tensor(self.mvns[i].logpdf(X))
        w = self.weights.reshape(n_modes, -1) * prob_factor
        log_p_X = utility.logsumexp(log_p_m_X, dim=0, b=w)
        return torch.as_tensor(log_p_X)

    def logpdf_explicit_unused(self, X):
        m = self.means[0]
        C = self.covs[0]
        C_inv = torch.inv(C)
        l_X = - ((X - m) @ C_inv * (X - m)).sum(axis=1) / 2
        const = (- len(m) * torch.log(2 * np.pi) - torch.log(torch.det(C))
                 ) / 2
        return l_X + const

    def cross_entropy(self, X):
        return -self.log_pdf(X).mean()

    def cross_entropy_sample(self, Q, n_samples=1e4):
        X = self.sample(n_samples)
        return -Q.log_pdf(X).mean()

    def DKL_sample(self, Q, n_samples=1e4):
        X = self.sample(n_samples)
        return self.log_pdf(X).mean() - Q.log_pdf(X).mean()

    def JS_sample(self, Q, n_samples=1e4, M=None):
        P = self
        if M is None:
            M = P + Q
        return 1 / 2 * (P.DKL_sample(M, n_samples) + Q.DKL_sample(M, n_samples))
