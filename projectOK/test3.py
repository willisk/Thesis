""" Testing reconstruction by matching statistics
on random projections
"""
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import datasets
import utility
import deepinversion

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(deepinversion)

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

X_A = torch.from_numpy(gmm.sample(n_samples=100))
mean_A = X_A.mean(dim=0)

# perturbed Dataset B
perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B_orig = torch.from_numpy(gmm.sample(n_samples=100))
X_B = X_B_orig @ perturb_matrix + perturb_shift

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


# plot random projections
print("Before:")
print("Cross Entropy of A:", gmm.cross_entropy(X_A))
print("Cross Entropy of B:", gmm.cross_entropy(X_B))
utility.plot_random_projections(RP, X_A, mean=mean_A)
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.legend()
plt.show()

utility.plot_random_projections(RP, X_B, mean=mean_A)
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="perturbed Data B")
plt.legend()
plt.show()


# ======= Preprocessing Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def preprocessing(X):
    return X @ A + b


def project(X):
    return (X - mean_A) @ RP


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

deepinversion.deep_inversion(X_B,
                             loss_fn,
                             optimizer,
                             steps=steps,
                             pre_fn=preprocessing,
                             )

# ======= Result =======
X_B_proc = preprocessing(X_B).detach()
print("After Pre-Processing:")
print("Cross Entropy of B:", gmm.cross_entropy(X_B_proc))
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="preprocessed Data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="unperturbed Data B", alpha=0.4)
utility.plot_stats([X_A, X_B_proc])
plt.legend()
plt.show()

print("effective transformation X.A + b")
print("A (should be close to Id):")
print((A @ perturb_matrix).detach())
print("b (should be close to 0):")
print((A @ perturb_shift + b).detach())
