"""Testing reconstruction by matching
class-conditional statistics
"""
import os
import sys

import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

from utils import utility
from utils import datasets

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)

print(__doc__)
cmaps = utility.categorical_colors(2)


# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
# Gaussian Mixture Model

n_classes = 2
dataset = datasets.MULTIGMM(
    n_dims=2,
    n_classes=n_classes,
    n_modes=5,
    scale_mean=6,
    scale_cov=3,
    mean_shift=20,
    n_samples_A=100,
    n_samples_B=100,
)

X_A, Y_A = dataset.A.tensors
X_B, Y_B = dataset.B.tensors
mean_A = X_A.mean(dim=0)

# distorted Dataset B
distort_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


X_B_orig, Y_B = X_B, Y_B
X_B = distort(X_B_orig)

# Y_B_orig = Y_B
# Y_A.fill_(0)
# Y_B = Y_B * 0

plt.scatter(X_A[Y_A == 0][:, 0], X_A[Y_A == 0][:, 1],
            c=cmaps[0], marker='+', alpha=0.4, label="target data A cl 0")
plt.scatter(X_A[Y_A == 1][:, 0], X_A[Y_A == 1][:, 1],
            c=cmaps[0], marker='d', alpha=0.4, label="target data A cl 1")
plt.scatter(X_B[Y_B == 0][:, 0], X_B[Y_B == 0][:, 1],
            c=cmaps[1], marker='+', alpha=0.4, label="data B cl 0")
plt.scatter(X_B[Y_B == 1][:, 0], X_B[Y_B == 1][:, 1],
            c=cmaps[1], marker='d', alpha=0.4, label="data B cl 1")
utility.plot_stats([X_A[Y_A == 0], X_B[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B[Y_B == 1]])
plt.legend()
plt.show()

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B).item())

# ======= reconstruct Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def reconstruct(X):
    return X @ A + b


# ======= Collect Stats from A =======
# collect stats
# shape: [n_class, n_dims] = [2, 2]
A_means, A_vars = utility.c_stats(X_A, Y_A, n_classes)


# ======= Loss Function =======
# def loss_frechet(X, Y=Y_B):
#     X_means, X_vars, _ = utility.c_mean_var(X, Y, n_classes)
#     diff_mean = ((X_means - A_means)**2).sum(dim=0).mean()
#     diff_var = (X_vars + A_vars - 2 * (X_vars * A_vars).sqrt()
#                 ).sum(dim=0).mean()
#     loss = (diff_mean + diff_var)
#     return loss


# def loss_fn(X, Y=Y_B):
#     # log likelihood * 2 - const:
#     # diff_mean = (((X_means - A_means)**2 / X_vars.detach())).sum(dim=0)
#     X_means, X_vars, _ = utility.c_mean_var(X, Y, n_classes)
#     diff_mean = ((X_means - A_means)**2)
#     diff_var = torch.abs(X_vars - A_vars).sqrt()
#     loss = diff_mean.mean() + diff_var.mean()
#     # loss = (0
#     #         + diff_mean[0].mean()
#     #         + diff_mean[1].mean()
#     #         + diff_var[0].mean()
#     #         + diff_var[1].mean()
#     #         )
#     return loss
def loss_fn(data):
    X, Y = data
    X = reconstruct(X)

    X_means, X_vars = utility.c_stats(X, Y, n_classes)
    loss_mean = (X_means - A_means).norm(dim=1).mean()
    loss_var = (X_vars - A_vars).norm(dim=1).mean()

    loss = loss_mean + loss_var

    info = {
        'loss': loss,
        '[losses] mean': loss_mean.item(),
        '[losses] var': loss_var.item(),
        'c-entropy': dataset.cross_entropy(X),
    }

    return info

# loss_fn = loss_frechet


# ======= Optimize =======
lr = 0.1
steps = 100
optimizer = torch.optim.Adam([A, b], lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, verbose=True)


utility.invert([(X_B, Y_B)],
               loss_fn,
               optimizer,
               #    scheduler=scheduler,
               steps=steps,
               plot=True,
               track_grad_norm=True,
               )

# x_history, steps = zip(*history)
# for x in x_history[30:]:
#     utility.plot_stats(x[Y_B == 0], colors=['r'] * len(history))

# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:",
      dataset.cross_entropy(X_B_orig).item())

plt.title("target data A")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', alpha=0.4, label="undistorted data B")
utility.plot_stats([X_A[Y_A == 0], X_B_proc[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B_proc[Y_B == 1]])
plt.legend()
plt.show()


# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")


# writer.close()
