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
import deepinversion

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(deepinversion)

print(__doc__)

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(1)
torch.manual_seed(1)

# ======= Create Dataset =======
# Gaussian Mixture Model

gmm = datasets.random_gmm(
    n_dims=2,
    n_modes=5,
    scale_mean=6,
    scale_cov=3,
    mean_shift=30,
)

X_A = gmm.sample(n_samples=100)
m_A, v_A = X_A.mean(dim=0), X_A.var(dim=0)

# perturbed Dataset B
perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)


def perturb(X):
    return X @ perturb_matrix + perturb_shift


X_B_orig = gmm.sample(n_samples=100)
X_B = perturb(X_B_orig)
m_B, v_B = X_B.mean(dim=0), X_B.var(dim=0)

print("Before:")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="perturbed Data B")
utility.plot_stats([X_A, X_B])
plt.legend()
plt.show()
print("Cross Entropy of A:", gmm.cross_entropy(X_A).item())
print("Cross Entropy of B:", gmm.cross_entropy(X_B).item())

# ======= Preprocessing Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def preprocessing(X):
    return X @ A + b


# ======= Loss Function =======
def loss_fn(X):
    # log likelihood * 2 - const
    # loss_mean = ((X - m_A)**2 / v_A.detach()).sum(dim=0).mean()
    loss_mean = ((X.mean(dim=0) - m_A)**2).mean()
    loss_var = ((X.var(dim=0) - v_A)**2).mean()
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
                             plot=True,
                             )

# ======= Result =======
X_B_proc = preprocessing(X_B).detach()
print("After Pre-Processing:")
print("Cross Entropy of B:", gmm.cross_entropy(X_B_proc).item())
print("Cross Entropy of unperturbed B:", gmm.cross_entropy(X_B_orig).item())
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="preprocessed Data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="unperturbed Data B", alpha=0.4)
utility.plot_stats([X_A, X_B_proc])
plt.legend()
plt.show()

m_B_pre, v_B_pre = X_B_proc.mean(dim=0), X_B_proc.var(dim=0)

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (preprocessing(perturb(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
