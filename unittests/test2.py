"""Testing reconstruction by matching statistics
of Gaussian Mixture Model in input space..

Comment:
Statistics are matched, but data is deformed.
Not enough information given.
"""
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import datasets

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)

print('#', __doc__)

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(0)
torch.manual_seed(0)

# ======= Create Dataset =======
# Gaussian Mixture Model

gmm = datasets.random_gmm(
    n_dims=2,
    n_modes=5,
    scale_mean=6,
    scale_cov=3,
    mean_shift=30,
)


def cov(X, mean=None):
    if mean is None:
        mean = X.mean(dim=0)
    return (X - mean).T @ (X - mean) / len(X)


X_A = gmm.sample(n_samples=100)
m_A, v_A = X_A.mean(dim=0), X_A.var(dim=0)
cov_A = cov(X_A)

# distorted Dataset B
distort_matrix = torch.eye(2) + 3 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


X_B_orig = gmm.sample(n_samples=100)
X_B = distort(X_B_orig)
m_B, v_B = X_B.mean(dim=0), X_B.var(dim=0)

print("Before:")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="distorted data B")
utility.plot_stats([X_A, X_B])
plt.legend()
plt.show()
print("Cross Entropy of A:", gmm.cross_entropy(X_A).item())
print("Cross Entropy of B:", gmm.cross_entropy(X_B).item())

# ======= reconstruct Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def reconstruct(X):
    return X @ A + b


# ======= Loss Function =======
def loss_fn(X):
    X = reconstruct(X)
    loss_mean = (X.mean(dim=0) - m_A).norm()
    loss_var = (X.var(dim=0) - v_A).norm()

    loss = loss_mean + loss_var
    info = {
        'loss': loss,
        '[losses] mean': loss_mean.item(),
        '[losses] var': loss_var.item(),
    }
    return info
    # return loss_mean + loss_var


# ======= Optimize =======
lr = 0.1
steps = 100
optimizer = torch.optim.Adam([A, b], lr=lr)

utility.invert([X_B],
               loss_fn,
               optimizer,
               steps=steps,
               plot=True,
               track_grad_norm=True,
               )

# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", gmm.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:", gmm.cross_entropy(X_B_orig).item())
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="undistorted data B", alpha=0.4)
utility.plot_stats([X_A, X_B_proc])
plt.legend()
plt.show()

m_B_pre, v_B_pre = X_B_proc.mean(dim=0), X_B_proc.var(dim=0)

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")


print("\nUsing Covariance Matrix:")
# ======= Loss Function =======


def loss_fn2(X):
    X = reconstruct(X)
    loss_mean = (X.mean(dim=0) - m_A).norm()
    loss_var = (cov(X, mean=X.mean(dim=0).detach()) - cov_A).norm()

    loss = loss_mean + loss_var
    info = {
        'loss': loss,
        '[losses] mean': loss_mean.item(),
        '[losses] var': loss_var.item(),
    }
    return info
    # return loss_mean + loss_var


# ======= Optimize =======
lr = 0.1
steps = 100
optimizer = torch.optim.Adam([A, b], lr=lr)

utility.invert([X_B],
               loss_fn2,
               optimizer,
               steps=steps,
               plot=True,
               track_grad_norm=True,
               )

# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", gmm.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:", gmm.cross_entropy(X_B_orig).item())
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="undistorted data B", alpha=0.4)
utility.plot_stats([X_A, X_B_proc])
plt.legend()
plt.show()

m_B_pre, v_B_pre = X_B_proc.mean(dim=0), X_B_proc.var(dim=0)

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
