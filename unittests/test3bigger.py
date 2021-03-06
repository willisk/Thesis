"""Test3 using random projections
on 2 distinct clusters
Comment:
Method struggles with more seperated clusters
"""
import os
import sys

import torch
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

dataset = datasets.MULTIGMM(
    n_dims=2,
    n_classes=2,
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

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


def project(X):
    return (X - mean_A) @ RP


# plot random projections
print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B).item())
utility.plot_random_projections(RP, project(X_A), mean=mean_A)
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.legend()
plt.show()

utility.plot_random_projections(RP, project(X_B), mean=mean_A)
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="distorted data B")
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


# ======= Collect Projected Stats from A =======
X_A_proj = project(X_A)


# collect stats
A_proj_means = X_A_proj.mean(dim=0, keepdims=True)
A_proj_vars = X_A_proj.var(dim=0, keepdims=True)


# ======= Loss Function =======
def loss_fn(X):
    X = reconstruct(X)
    X_proj = project(X)
    # loss_mean = ((X_proj.mean(dim=0) - A_proj_means)**2).mean()
    # loss_var = ((X_proj.var(dim=0) - A_proj_vars)**2).mean()
    # return loss_mean + loss_var

    loss_mean = (X_proj.mean(dim=0) - A_proj_means).norm()
    loss_var = (X_proj.var(dim=0) - A_proj_vars).norm()

    loss = loss_mean + loss_var

    info = {
        'loss': loss,
        '[losses] mean': loss_mean.item(),
        '[losses] var': loss_var.item(),
        'c-entropy': dataset.cross_entropy(X),
    }

    return info


# ======= Optimize =======
lr = 0.1
steps = 300
optimizer = torch.optim.Adam([A, b], lr=lr)

info = utility.invert([X_B],
                      loss_fn,
                      optimizer,
                      steps=steps,
                      plot=True,
                      track_grad_norm=True,
                      )

# for x, step in zip(*zip(*history)):
#     utility.plot_stats(x, colors=['r'] * len(history))
# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:",
      dataset.cross_entropy(X_B_orig).item())
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="undistorted data B", alpha=0.4)
utility.plot_stats([X_A[Y_A == 0], X_B_proc[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B_proc[Y_B == 1]])
plt.legend()
plt.show()


utility.plot_random_projections(
    RP, project(X_A), mean=mean_A, color=cmaps[0], scatter=True)
plt.scatter(X_A[:, 0], X_A[:, 1],
            c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="distorted data B")
plt.legend(loc="lower right")
plt.show()
utility.plot_random_projections(
    RP, project(X_B_proc), mean=mean_A, color=cmaps[1], scatter=True)
plt.scatter(X_A[:, 0], X_A[:, 1],
            c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="distorted data B")
plt.legend()
plt.show()

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
