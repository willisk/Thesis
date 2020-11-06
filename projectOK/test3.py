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
import statsnet
import utility
import deepinversion
import shared

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)


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

perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B_orig = torch.from_numpy(gmm.sample(n_samples=100))
X_B = X_B_orig.matmul(perturb_matrix) + perturb_shift

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


# plot random projections
print("Before:")
print("Cross Entropy of A:", gmm.cross_entropy(X_A))
print("Cross Entropy of B:", gmm.cross_entropy(X_B))
cmaps = utility.categorical_colors(2)
utility.plot_random_projections(RP, X_A)
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.legend()
# plt.axis('equal')
plt.show()

utility.plot_random_projections(RP, X_B)
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="perturbed Data B")
plt.legend()
plt.show()


# ======= Preprocessing Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def preprocessing(X):
    return X.matmul(A) + b


# ======= Collect Projected Stats from A =======
# project
X_A_proj = X_A.matmul(RP)

# collect stats
A_proj_means = X_A_proj.mean(dim=0, keepdims=True)
A_proj_vars = X_A_proj.var(dim=0, keepdims=True)


# ======= Loss Function =======
def loss_fn(X):
    X_proj = X.matmul(RP)
    loss_mean = ((X_proj.mean(dim=0) - A_proj_means)**2).mean()
    loss_var = ((X_proj.var(dim=0) - A_proj_vars)**2).mean()
    return loss_mean + loss_var


# ======= Optimize =======
lr = 0.1
steps = 400
optimizer = torch.optim.Adam([A, b], lr=lr)

invert = deepinversion.deep_inversion(X_B,
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

net_matrix = perturb_matrix.matmul(A).detach()
net_shift = (perturb_shift.matmul(A) + b).detach()
print("effective transform matrix: (should be close to Id)")
print(net_matrix)
print("effective shift: (should be close to 0)")
print(net_shift)
