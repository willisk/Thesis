import os
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(PWD, "models")
DATADIR = os.path.join(PWD, "data")

sys.path.append(PWD)

import numpy as np
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, TensorDataset, Subset
import torchvision.transforms as T

# import matplotlib
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.pyplot as plt

# https://github.com/huyvnphan/PyTorch_CIFAR10
# from ext.cifar10pretrained.cifar10_models import resnet34, resnet50
# from torchvision.models import resnet34 as ResNet34, resnet50 as ResNet50
# from ext.cifar10pretrained.cifar10_download import main as download_resnet
# download_resnet()
from .debug import debug
from . import utility
from . import nets

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(utility)
    importlib.reload(nets)


def split_dataset(dataset, split=0.8):
    n_a = int(len(dataset) * split)
    n_b = len(dataset) - n_a
    return random_split(dataset, (n_a, n_b))


class Dataset():
    def __init__(self, input_shape, n_classes, A, B, C=None,
                 data_dir=MODELDIR, transform=None):
        self.A = A
        self.B = B
        self.C = C

        self.data_dir = data_dir

        self.input_shape = input_shape
        self.n_dims = np.prod(input_shape)
        self.n_classes = n_classes

        self.transform = transform

    def get_datasets(self, size_A=-1, size_B=-1, size_C=-1):
        size_A = min(size_A, len(self.A)) if size_A != -1 else len(self.A)
        size_B = min(size_B, len(self.B)) if size_B != -1 else len(self.B)
        size_C = min(size_C, len(self.C)) if size_C != -1 else len(self.C)

        A = Subset(self.A, torch.randperm(len(self.A))[:size_A])
        B = Subset(self.B, torch.randperm(len(self.B))[:size_B])
        C = Subset(self.C, torch.randperm(len(self.C))[:size_C])
        return A, B, C

    def net(self):
        return None, None

    def verification_net(self):
        return None, None


class CIFAR10(Dataset):
    def __init__(self, load_data=True):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
        ])
        if load_data:
            train_set = torchvision.datasets.CIFAR10(
                root=DATADIR, train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(
                root=DATADIR, train=False, download=True, transform=transform)

            A, B = split_dataset(train_set, split=0.8)
            C = test_set
        else:
            A, B, C = None, None, None

        data_dir = os.path.join(MODELDIR, "CIFAR10")

        super().__init__(input_shape=(3, 32, 32),
                         n_classes=10,
                         A=A,
                         B=B,
                         C=C,
                         data_dir=data_dir,
                         transform=transform)

    def net(self):
        resnet = nets.resnet34()
        model_path = os.path.join(self.data_dir, "net_resnet34.pt")
        return model_path, resnet

    def verification_net(self):
        resnet = nets.resnet18()
        model_path = os.path.join(self.data_dir, "net_resnet18.pt")
        return model_path, resnet


class MNIST(Dataset):
    def __init__(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(
            root=DATADIR, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root=DATADIR, train=False, download=True, transform=transform)

        A, B = split_dataset(train_set, split=0.8)
        C = test_set

        data_dir = os.path.join(MODELDIR, "MNIST")

        super().__init__(input_shape=(1, 28, 28),
                         n_classes=10,
                         A=A,
                         B=B,
                         C=C,
                         data_dir=data_dir,
                         transform=transform)

    def net(self):
        resnet = nets.ResNet20(1, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet20.pt")
        return model_path, resnet

    def verification_net(self):
        resnet = nets.ResNet20(1, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet20_ver.pt")
        return model_path, resnet


class SVHN(Dataset):
    def __init__(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4377, 0.4438, 0.4728], [0.1201, 0.1231, 0.1052])
        ])
        train_set = torchvision.datasets.SVHN(
            root=DATADIR, split='train', download=True, transform=transform)
        test_set = torchvision.datasets.SVHN(
            root=DATADIR, split='test', download=True, transform=transform)

        A, B = split_dataset(train_set, split=0.8)
        C = test_set

        data_dir = os.path.join(MODELDIR, "SVHN")

        super().__init__(input_shape=(3, 32, 32),
                         n_classes=10,
                         A=A,
                         B=B,
                         C=C,
                         data_dir=data_dir,
                         transform=transform)

    def net(self):
        resnet = nets.ResNet20(3, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet20.pt")
        return model_path, resnet

    def verification_net(self):
        resnet = nets.ResNet20(3, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet20_ver.pt")
        return model_path, resnet


# ========= GMMs ==========
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, ortho_group


# if args.dataset == 'GMM':
#     dataset = datasets.MULTIGMM(
#         input_shape=(20,),
#         n_classes=3,
#         n_modes=args.g_modes,
#         scale_mean=args.g_scale_mean,
#         scale_cov=args.g_scale_cov,
#         mean_shift=args.g_mean_shift,
#         n_samples_A=1000,
#         n_samples_B=100,
#         n_samples_C=100,
#     )
class MULTIGMM(Dataset):

    def __init__(self,
                 #  n_dims=20,
                 n_dims=2,
                 n_classes=3,
                 n_modes=3,
                 #  n_classes=10,
                 #  n_modes=12,
                 scale_mean=12,
                 scale_cov=25,
                 mean_shift=36,
                 n_samples_A=1000,
                 #  n_samples_A=10000,
                 n_samples_B=2000,
                 n_samples_C=2000
                 ):

        # # using equal weights for now
        # if weights is None:
        #     weights = torch.ones((n_classes))
        # self.weights = weights / weights.sum()

        self.gmms = [
            random_gmm(n_modes, n_dims, scale_mean, scale_cov, mean_shift)
            for _ in range(n_classes)
        ]

        A = self.sample(int(n_samples_A / n_classes))
        B = self.sample(int(n_samples_B / n_classes))
        C = self.sample(int(n_samples_C / n_classes))

        mean = A[0].mean()
        std = A[0].std()

        # A = (A[0].reshape(-1, 2, 5, 2), A[1])
        # B = (B[0].reshape(-1, 2, 5, 2), B[1])
        # C = (C[0].reshape(-1, 2, 5, 2), C[1])

        A = TensorDataset(*A)
        B = TensorDataset(*B)
        C = TensorDataset(*C)

        data_dir = os.path.join(MODELDIR, "GMM")

        def normalize(X):
            return (X - mean) / std

        super().__init__(input_shape=(n_dims,),
                         n_classes=n_classes,
                         A=A,
                         B=B,
                         C=C,
                         data_dir=data_dir,
                         transform=normalize)

    def net(self, suffix=""):
        nn_width = 32
        nn_depth = 4
        nn_layer_dims = [self.n_dims] + \
            [nn_width] * nn_depth + [self.n_classes]
        fcnet = nets.FCNet(nn_layer_dims, batch_norm=True)
        model_name = f"net_GMM_{'-'.join(map(repr, nn_layer_dims))}{suffix}.pt"
        model_path = os.path.join(self.data_dir, model_name)
        return model_path, fcnet

    def verification_net(self):
        return self.net(suffix="_verification")

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

    def pdf(self, X):
        return torch.exp(self.log_pdf(X))

    @torch.no_grad()
    def log_pdf(self, X):
        n_class = len(self.gmms)
        logpX_C_M, b_C = [None] * n_class, [None] * n_class
        a_max = -float('inf')
        n = len(X)
        for c, gmm in enumerate(self.gmms):
            n_modes = len(gmm.means)
            logpX_C_M[c] = torch.empty((n_modes, n))
            for m in range(n_modes):
                logpX_C_M[c][m] = torch.as_tensor(gmm.mvns[m].logpdf(X.cpu()))
                # logpX_C_M[c][m] = gmm.logpdf_explicit_unused(X)
            a_max = max(a_max, torch.max(logpX_C_M[c]).item())
        p_X = torch.zeros((len(X)))
        for c in range(n_class):
            weights = gmm.weights.reshape(n_modes, 1)
            p_X += (weights * torch.exp(logpX_C_M[c] - a_max)).mean(dim=0)
        return torch.log(p_X) + a_max

    def cross_entropy(self, X, Y=None):
        return -self.log_pdf(X).mean()

    def plot(self):
        assert self.n_dims == 2
        X, Y = self.A.tensors
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=3)
        offs = torch.Tensor([2, 1])
        plt.scatter([0], [0], c='k', s=5)
        plt.annotate('0', offs, c='k')
        for c, gmm in enumerate(self.gmms):
            if c != 2:
                continue
            debug(gmm.means)
            center = torch.mean(gmm.means, axis=0)
            plt.plot((0, center[0]), (0, center[1]), 'k:', lw=1, zorder=0)
            plt.annotate('$\\gamma$', (center * 0.5 + offs))
            plt.scatter(*center, c='red')
            plt.annotate(f'$m_1$', center + offs, c='red',
                         bbox={'facecolor': 'white', 'edgecolor': 'white', 'pad': 0}, zorder=1)
            for m, mean in enumerate(gmm.means):
                if m == 1:
                    plt.plot((center[0], mean[0]),
                             (center[1], mean[1]), 'k:', lw=1, zorder=0)
                    plt.annotate(
                        '$\\lambda$', (center + (mean - center) * 0.5 + offs), zorder=1)
                plt.scatter(*mean, marker='^', c='red', zorder=1)
                plt.annotate(
                    f'$\\mu_1^{{({m + 1})}}$', mean + offs, c='red', bbox={'facecolor': 'white', 'edgecolor': 'white', 'pad': 0}, zorder=1)
        # plt.scatter(X[:, 0], X[:, 1])


def make_spd_matrix(n_dims, eps_min=0.3):
    D = torch.diag(torch.rand((n_dims)) + eps_min)
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

    # def pairwise_cross_entropy(self):
    #     if len(self.gmms) < 2:
    #         return
    #     print("0 < H < inf")
    #     for i, P in enumerate(self.gmms):
    #         Q_rest = combine_gmms([Q for Q in self.gmms if Q != P])
    #         print(f"H(P{i}, Q_rest) = {P.cross_entropy_sample(Q_rest)}")

    # def pairwise_JS(self):
    #     if len(self.gmms) < 2:
    #         return
    #     print("0 < JS < ln(2)")
    #     M = combine_gmms(self.gmms)
    #     for i, P in enumerate(self.gmms):
    #         Q_rest = combine_gmms([Q for Q in self.gmms if Q != P])
    #         print(f"JS(P{i}|Q_rest) = {P.JS_sample(Q_rest, M=M)}")


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
            log_p_m_X[i] = torch.as_tensor(self.mvns[i].logpdf(X.detach()))
        w = self.weights.reshape(n_modes, 1) * prob_factor
        log_p_X = utility.logsumexp(log_p_m_X, dim=0, b=w)
        return torch.as_tensor(log_p_X)

    def logpdf_explicit_unused(self, X):
        if not hasattr(self, 'C_inv'):
            self.C_inv = [torch.inverse(C.double()) for C in self.covs]
            self.log_det_C = [torch.log(torch.det(C.double()))
                              for C in self.covs]
        n_modes, n_samples = len(self.means), len(X)
        d = X.shape[1]
        logpX_M = torch.empty((n_modes, n_samples))
        B = [(X.double() - m.double()) for m in self.means]
        for m in range(n_modes):
            logpX_M[m] = ((B[m] @ self.C_inv[m] * B[m]).sum(axis=1)
                          + d * np.log(2 * np.pi) + self.log_det_C[m]) / -2
        logpX = utility.logsumexp(
            logpX_M, dim=0, b=self.weights.reshape(n_modes, 1))
        return logpX

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
