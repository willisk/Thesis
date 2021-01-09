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

# https://github.com/huyvnphan/PyTorch_CIFAR10
from ext.cifar10pretrained.cifar10_models import resnet34, resnet50
# from torchvision.models import resnet34 as ResNet34, resnet50 as ResNet50
# from ext.cifar10pretrained.cifar10_download import main as download_resnet
# download_resnet()
import utility
import nets

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(utility)
    importlib.reload(nets)


def split_dataset(dataset, split=0.8):
    n_a = int(len(dataset) * split)
    n_b = len(dataset) - n_a
    return random_split(dataset, (n_a, n_b))


class Dataset():
    def __init__(self, input_shape, n_classes, A, B, B_val=None,
                 data_dir=MODELDIR, transform=None):
        self.A = A
        self.B = B
        self.B_val = B_val

        self.data_dir = data_dir

        self.input_shape = input_shape
        self.n_dims = np.prod(input_shape)
        self.n_classes = n_classes

        self.transform = transform

    def get_datasets(self, size_A=-1, size_B=-1, size_B_val=-1):
        size_A = min(size_A, len(self.A)) if size_A != -1 else len(self.A)
        size_B = min(size_B, len(self.B)) if size_B != -1 else len(self.B)
        size_B_val = min(size_B_val, len(self.B_val)
                         ) if size_B_val != -1 else len(self.B_val)

        A = Subset(self.A, torch.randperm(len(self.A))[:size_A])
        B = Subset(self.B, torch.randperm(len(self.B))[:size_B])
        B_val = Subset(self.B_val, torch.randperm(
            len(self.B_val))[:size_B_val])
        return A, B, B_val

    def net(self):
        return None, None

    def verifier_net(self):
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
            B_val = test_set
        else:
            A, B, B_val = None, None, None

        data_dir = os.path.join(MODELDIR, "CIFAR10")

        super().__init__(input_shape=(3, 32, 32),
                         n_classes=10,
                         A=A,
                         B=B,
                         B_val=B_val,
                         data_dir=data_dir,
                         transform=transform)

    def net(self):
        resnet = nets.resnet34()
        model_path = os.path.join(self.data_dir, "net_resnet34.pt")
        return model_path, resnet

    def verifier_net(self):
        resnet = nets.resnet50()
        model_path = os.path.join(self.data_dir, "net_resnet50.pt")
        return model_path, resnet


class MNIST(Dataset):
    def __init__(self):
        transform = torchvision.T.Compose([
            torchvision.T.ToTensor(),
            torchvision.T.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(
            root=DATADIR, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(
            root=DATADIR, train=False, download=True, transform=transform)

        A, B = split_dataset(train_set, split=0.8)
        B_val = test_set

        data_dir = os.path.join(MODELDIR, "MNIST")

        super().__init__(input_shape=(1, 28, 28),
                         n_classes=10,
                         A=A,
                         B=B,
                         B_val=B_val,
                         data_dir=data_dir,
                         transform=transform)

    def net(self):
        resnet = nets.ResNet20(1, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet20.pt")
        return model_path, resnet

    def verifier_net(self):
        resnet = nets.ResNet9(1, self.n_classes)
        model_path = os.path.join(self.data_dir, "net_resnet9.pt")
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
#         n_samples_B_val=100,
#     )
class MULTIGMM(Dataset):

    def __init__(self, input_shape=20, n_classes=10, n_modes=8,
                 scale_mean=1, scale_cov=1, mean_shift=0,
                 n_samples_A=1000,
                 n_samples_B=1000,
                 n_samples_B_val=1000,
                 device='cpu', weights=None):

        # # using equal weights for now
        # if weights is None:
        #     weights = torch.ones((n_classes))
        # self.weights = weights / weights.sum()
        self.device = device

        self.gmms = [
            random_gmm(n_modes, n_dims, scale_mean, scale_cov, mean_shift)
            for _ in range(n_classes)
        ]

        A = self.sample(int(n_samples_A / n_classes))
        B = self.sample(int(n_samples_B / n_classes))
        B_val = self.sample(int(n_samples_B_val / n_classes))

        # A = (A[0].reshape(-1, 2, 5, 2), A[1])
        # B = (B[0].reshape(-1, 2, 5, 2), B[1])
        # B_val = (B_val[0].reshape(-1, 2, 5, 2), B_val[1])

        A = TensorDataset(*A)
        B = TensorDataset(*B)
        B_val = TensorDataset(*B_val)

        data_dir = os.path.join(MODELDIR, "GMM")

        super().__init__(n_dims=n_dims,
                         n_classes=n_classes,
                         A=A,
                         B=B,
                         B_val=B_val,
                         data_dir=data_dir,
                         transform=None)

    def net(self, suffix=""):
        nn_width = 32
        nn_depth = 4
        nn_layer_dims = [self.n_dims] + \
            [nn_width] * nn_depth + [self.n_classes]
        fcnet = nets.FCNet(nn_layer_dims)
        model_name = f"net_GMM_{'-'.join(map(repr, nn_layer_dims))}{suffix}"
        model_path = os.path.join(self.data_dir, model_name)
        return model_path, fcnet

    def verifier_net(self):
        return self.net(suffix="_verifier")

    def sample(self, n_samples_per_class):
        X = Y = None
        for c, gmm in enumerate(self.gmms):
            x = gmm.sample(n_samples_per_class)
            y = torch.LongTensor([c] * n_samples_per_class)
            X = torch.cat((X, x), dim=0) if X is not None else x
            Y = torch.cat((Y, y), dim=0) if Y is not None else y
        return X.to(self.device), Y.to(self.device)

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
            # estimated class prob, should use equal?
            p_c = (Y == c).sum().item() / len(Y)
            n_modes, n_c = len(gmm.means), len(X_c)
            b[c] = gmm.weights.reshape(n_modes, -1) * p_c
            a[c] = torch.empty((n_modes, n_c))
            for m in range(n_modes):
                a[c][m] = torch.as_tensor(gmm.mvns[m].logpdf(X_c))
            a_max = max(a_max, torch.max(a[c]).item())
        p_X = torch.zeros((len(X)))
        for c, (log_p_X_c_m, w) in enumerate(zip(a, b)):
            if log_p_X_c_m is None:
                continue
            p_X_c = (w * torch.exp(log_p_X_c_m - a_max)).sum(dim=0)
            if Y is not None:
                p_X[Y == c] = p_X_c
            else:
                p_X = p_X + p_X_c
        # print((p_X == 0).sum().item())
        # print(p_X.shape)
        return torch.log(p_X) + a_max

    def cross_entropy(self, X, Y=None):
        X = X.to('cpu')
        if Y is not None:
            Y = Y.to('cpu')
        return -self.log_pdf(X, Y=Y).mean()

    # def concatenate(self, except_for=None):
    #     gmms = [g for g in self.gmms if g is not except_for]
    #     means, covs, weights = zip(
    #         *[(q.means, q.covs, q.weights) for q in gmms])
    #     means = torch.cat(means, dim=0)
    #     covs = torch.cat(covs, dim=0)
    #     weights = torch.cat(weights, dim=0)
    #     weights /= weights.sum()
    #     return GMM(means, covs, weights)


def make_spd_matrix(n_dims, eps_min=0.3):
    D = torch.diag(torch.abs(torch.rand((n_dims)))) + eps_min
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
