""" Testing reconstruction by matching 
class-conditional statistics on random projections
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

dataset = datasets.DatasetGMM(
    n_dims=2,
    n_modes=5,
    scale_mean=6,
    scale_cov=3,
    mean_shift=20,
    n_classes=2,
    n_samples_per_class=100,
)

# plot dataset
X_A, Y_A = dataset.X, dataset.Y
plt.title("Data A")
dataset.plot()
utility.plot_stats(X_A, Y_A)
plt.show()

perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B_orig, Y_B = dataset.sample(n_samples_per_class=100)
X_B = X_B_orig.matmul(perturb_matrix) + perturb_shift

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


# plot random projections
cmaps = utility.categorical_colors(2)
print("Before:")
plt.title("Data A")
utility.plot_random_projections(RP, X_A, Y_A)
plt.scatter(X_A[:, 0], X_A[:, 1], c=Y_A.squeeze(), cmap='Spectral', alpha=0.4)
plt.legend()
plt.show()
print("Cross Entropy of A:", dataset.cross_entropy(X_A, Y_A))

plt.title("perturbed Data B")
utility.plot_random_projections(RP, X_B, Y_B)
plt.scatter(X_B[:, 0], X_B[:, 1], c=Y_B.squeeze(), cmap='Spectral', alpha=0.4)
plt.legend()
plt.show()
print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B))


# ======= Preprocessing Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def preprocessing(X):
    return X.matmul(A) + b


# ======= Collect Projected Stats from A =======
# project
X_A_proj = X_A.matmul(RP)

# collect stats
# shape: [n_class, n_dims] = [2, 2]
# A_proj_means, A_proj_vars, _ = utility.c_mean_var(X_A_proj, Y_A)

A_proj_means = X_A_proj[Y_A == 0].mean(dim=0, keepdims=True)
A_proj_vars = X_A_proj[Y_A == 0].var(dim=0, keepdims=True)


# ======= Loss Function =======
def loss_fn(X, Y=Y_B):
    X_proj = X.matmul(RP)
    # X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y)
    X_proj_means = X_proj[Y == 0].mean(dim=0)
    X_proj_vars = X_proj[Y == 0].var(dim=0)
    loss_mean = ((X_proj_means - A_proj_means)**2).mean()
    loss_var = ((X_proj_vars - A_proj_vars)**2).mean()
    loss_mean = ((X_proj_means - A_proj_means)**2).mean()
    loss_var = ((X_proj_vars - A_proj_means)**2).mean()
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
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc, Y_B))
plt.title("Data A")
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
