"""Testing reconstruction by matching statistics
of Gaussian Mixture Model on random projections..
"""
import os
import sys

import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import datasets
import utility
import inversion

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(inversion)

print(__doc__)

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
# Gaussian Mixture Model

gmm = datasets.random_gmm(
    n_dims=2,
    n_modes=5,
    scale_mean=7,
    scale_cov=3,
    mean_shift=20,
)

X_A = gmm.sample(n_samples=100)
mean_A = X_A.mean(dim=0)

# distorted Dataset B
distort_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


X_B_orig = gmm.sample(n_samples=100)
X_B = distort(X_B_orig)

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


def project(X):
    return (X - mean_A) @ RP


# plot random projections
print("Before:")
print("Cross Entropy of A:", gmm.cross_entropy(X_A).item())
print("Cross Entropy of B:", gmm.cross_entropy(X_B).item())
utility.plot_random_projections(RP, project(X_A), mean=mean_A)
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.legend()
plt.show()

utility.plot_random_projections(RP, project(X_B), mean=mean_A)
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="distorted Data B")
plt.legend()
plt.show()


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
    X_proj = project(X)
    loss_mean = ((X_proj.mean(dim=0) - A_proj_means)**2).mean()
    loss_var = ((X_proj.var(dim=0) - A_proj_vars)**2).mean()
    return loss_mean + loss_var


# ======= Optimize =======
lr = 0.1
steps = 400
optimizer = torch.optim.Adam([A, b], lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, verbose=True)


def grad_norm_fn(x):
    return max(x, 1)


    # return np.sqrt(x) if x > 1 else x
inversion.deep_inversion([X_B],
                         loss_fn,
                         optimizer,
                         #  scheduler=scheduler,
                         steps=steps,
                         data_pre_fn=reconstruct,
                         #  grad_norm_fn=grad_norm_fn,
                         plot=True,
                         )

# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Pre-Processing:")
print("Cross Entropy of B:", gmm.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:", gmm.cross_entropy(X_B_orig).item())
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="preprocessed Data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="undistorted Data B", alpha=0.4)
utility.plot_stats([X_A, X_B_proc])
plt.legend()
plt.show()

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
