"""Testing reconstruction by matching
class-conditional statistics on random projections
Comment:
Similar to last experiment, no full convergence
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

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(deepinversion)

print(__doc__)

cmaps = utility.categorical_colors(2)


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

X_A, Y_A = dataset.X, dataset.Y
mean_A = X_A.mean(dim=0)

# perturbed Dataset B
perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B_orig, Y_B = dataset.sample(n_samples_per_class=100)
X_B = X_B_orig @ perturb_matrix + perturb_shift

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)

# plot random projections
plt.scatter(X_A[Y_A == 0][:, 0], X_A[Y_A == 0][:, 1],
            c=cmaps[0], marker='+', alpha=0.4, label="Data A cl 0")
plt.scatter(X_A[Y_A == 1][:, 0], X_A[Y_A == 1][:, 1],
            c=cmaps[0], marker='d', alpha=0.4, label="Data A cl 1")
plt.scatter(X_B[Y_B == 0][:, 0], X_B[Y_B == 0][:, 1],
            c=cmaps[1], marker='+', alpha=0.4, label="Data B cl 0")
plt.scatter(X_B[Y_B == 1][:, 0], X_B[Y_B == 1][:, 1],
            c=cmaps[1], marker='d', alpha=0.4, label="Data B cl 1")
utility.plot_stats([X_A[Y_A == 0], X_B[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B[Y_B == 1]])
plt.legend()
plt.show()

plt.title("Data A")
utility.plot_random_projections(RP, X_A, mean=mean_A, Y=Y_A, marker='+')
plt.scatter(X_A[Y_A == 0][:, 0], X_A[Y_A == 0][:, 1],
            c=cmaps[0], marker='+', alpha=0.4, label="Data A cl 0")
plt.scatter(X_A[Y_A == 1][:, 0], X_A[Y_A == 1][:, 1],
            c=cmaps[0], marker='d', alpha=0.4, label="Data A cl 1")
plt.legend()
plt.show()

plt.title("perturbed Data B")
utility.plot_random_projections(RP, X_B, mean=mean_A, Y=Y_B)
plt.scatter(X_B[Y_B == 0][:, 0], X_B[Y_B == 0][:, 1],
            c=cmaps[1], marker='+', alpha=0.4, label="Data B cl 0")
plt.scatter(X_B[Y_B == 1][:, 0], X_B[Y_B == 1][:, 1],
            c=cmaps[1], marker='d', alpha=0.4, label="Data B cl 1")
plt.legend()
plt.show()

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A, Y_A))
print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B))

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
# shape: [n_class, n_dims] = [2, 2]
A_proj_means, A_proj_vars, _ = utility.c_mean_var(X_A_proj, Y_A)


# ======= Loss Function =======
def loss_frechet(X, Y=Y_B):
    X_proj = project(X)
    X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y)
    diff_mean = ((X_proj_means - A_proj_means)**2).sum(dim=0).mean()
    diff_var = (X_proj_vars + A_proj_vars
                - 2 * (X_proj_vars * A_proj_vars).sqrt()
                ).sum(dim=0).mean()
    loss = (diff_mean + diff_var)
    return loss


def loss_fn(X, Y=Y_B):
    X_proj = project(X)
    X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y)
    loss_mean = ((X_proj_means - A_proj_means)**2).mean()
    loss_var = ((X_proj_vars - A_proj_vars)**2).mean()
    return loss_mean + loss_var


loss_fn = loss_frechet

# ======= Optimize =======
lr = 0.1
steps = 400
optimizer = torch.optim.Adam([A, b], lr=lr)

history = deepinversion.deep_inversion(X_B,
                                       loss_fn,
                                       optimizer,
                                       steps=steps,
                                       pre_fn=preprocessing,
                                       #    track_history=True,
                                       #    track_history_every=10,
                                       plot=True,
                                       )

# for x, step in zip(*zip(*history)):
#     utility.plot_stats(x, colors=['r'] * len(history))
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
utility.plot_stats([X_A[Y_A == 0], X_B_proc[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B_proc[Y_B == 1]])
plt.legend()
plt.show()


print("effective transformation X.A + b")
print("A (should be close to Id):")
print((A @ perturb_matrix).detach())
print("b (should be close to 0):")
print((A @ perturb_shift + b).detach())
