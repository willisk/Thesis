""" Testing reconstruction by matching statistics
in input space
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
    scale_mean=6,
    scale_cov=3,
    mean_shift=30,
)

X_A = torch.from_numpy(gmm.sample(n_samples=100))
m_A, v_A = X_A.mean(dim=0), X_A.var(dim=0)

perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B_orig = torch.from_numpy(gmm.sample(n_samples=100))
X_B = X_B_orig.matmul(perturb_matrix) + perturb_shift
m_B, v_B = X_B.mean(dim=0), X_B.var(dim=0)

print("Before:")
cmaps = utility.categorical_colors(2)
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="perturbed Data B")
utility.plot_stats([X_A, X_B])
plt.legend()
plt.show()

# ======= Preprocessing Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def preprocessing(X):
    return X.matmul(A) + b


# ======= Loss Function =======

def loss_fn(X):
    loss_mean = ((X.mean(dim=0) - m_A)**2).mean()
    loss_var = ((X.var(dim=0) - v_A)**2).mean()
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
print("After:")
X_B_pre = preprocessing(X_B).detach()
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B_pre[:, 0], X_B_pre[:, 1],
            c=cmaps[1], label="preprocessed Data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="unperturbed Data B", alpha=0.4)
utility.plot_stats([X_A, X_B_pre])
plt.legend()
plt.show()

m_B_pre, v_B_pre = X_B_pre.mean(dim=0), X_B_pre.var(dim=0)

print("net transform matrix (should be Id)")
print(A.matmul(perturb_matrix).detach())
print("net shift (should be 0)")
print((A.matmul(perturb_shift) + b).detach())
