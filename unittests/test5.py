"""Testing reconstruction by matching
class-conditional statistics on random projections
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
# np.random.seed(345)
# torch.manual_seed(346)
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
# Gaussian Mixture Model

n_classes = 2

dataset = datasets.DatasetGMM(
    n_dims=2,
    n_modes=5,
    scale_mean=6,
    scale_cov=3,
    mean_shift=20,
    n_classes=n_classes,
    n_samples_per_class=100,
)


X_A, Y_A = dataset.X, dataset.Y
# mean_A = X_A.mean(dim=0)
means_A, _, _ = utility.c_mean_var(X_A, Y_A, n_classes)

# distorted Dataset B
distort_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


X_B_orig, Y_B = dataset.sample(n_samples_per_class=100)
X_B = distort(X_B_orig)

# ======= Random Projections =======
n_projections = 3
RP = torch.randn((2, n_projections))
RP = RP / RP.norm(2, dim=0)


def project(X, Y):
    X_proj_C = torch.empty((X.shape[0], n_projections))
    for c in range(n_classes):
        X_proj_C[Y == c] = (X[Y == c] - means_A[c]) @ RP
    return X_proj_C

# # plot random projections
# plt.scatter(X_A[Y_A == 0][:, 0], X_A[Y_A == 0][:, 1],
#             c=cmaps[0], marker='+', alpha=0.4, label="target data A cl 0")
# plt.scatter(X_A[Y_A == 1][:, 0], X_A[Y_A == 1][:, 1],
#             c=cmaps[0], marker='d', alpha=0.4, label="target data A cl 1")
# plt.scatter(X_B[Y_B == 0][:, 0], X_B[Y_B == 0][:, 1],
#             c=cmaps[1], marker='+', alpha=0.4, label="data B cl 0")
# plt.scatter(X_B[Y_B == 1][:, 0], X_B[Y_B == 1][:, 1],
#             c=cmaps[1], marker='d', alpha=0.4, label="data B cl 1")
# utility.plot_stats([X_A[Y_A == 0], X_B[Y_B == 0]])
# utility.plot_stats([X_A[Y_A == 1], X_B[Y_B == 1]])
# plt.legend()

# plt.show()


plt.title("target data A")
utility.plot_random_projections(
    RP, project(X_A, Y_A), mean=means_A, Y=Y_A, marker='+')
plt.scatter(X_A[Y_A == 0][:, 0], X_A[Y_A == 0][:, 1],
            c=cmaps[0], marker='+', alpha=0.4, label="target data A cl 0")
plt.scatter(X_A[Y_A == 1][:, 0], X_A[Y_A == 1][:, 1],
            c=cmaps[0], marker='d', alpha=0.4, label="target data A cl 1")
plt.axis('equal')
plt.legend()
plt.show()

plt.title("distorted data B")
utility.plot_random_projections(RP, project(X_B, Y_A), mean=means_A, Y=Y_B)
plt.scatter(X_B[Y_B == 0][:, 0], X_B[Y_B == 0][:, 1],
            c=cmaps[1], marker='+', alpha=0.4, label="data B cl 0")
plt.scatter(X_B[Y_B == 1][:, 0], X_B[Y_B == 1][:, 1],
            c=cmaps[1], marker='d', alpha=0.4, label="data B cl 1")
plt.legend()
plt.axis('equal')
plt.show()

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A, Y_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B).item())

# ======= reconstruct Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def reconstruct(X):
    return X @ A + b


# ======= Collect Projected Stats from A =======
X_A_proj = project(X_A, Y_A)


# collect stats
# shape: [n_class, n_dims] = [2, 2]
A_proj_means, A_proj_vars, _ = utility.c_mean_var(X_A_proj, Y_A, n_classes)


# ======= Loss Function =======
def loss_frechet(X, Y=Y_B):
    X_proj = project(X, Y)
    X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y, n_classes)
    diff_mean = ((X_proj_means - A_proj_means)**2).sum(dim=0).mean()
    diff_var = (X_proj_vars + A_proj_vars
                - 2 * (X_proj_vars * A_proj_vars).sqrt()
                ).sum(dim=0).mean()
    loss = (diff_mean + diff_var)
    return loss


def loss_fn(X, Y=Y_B):
    X_proj = project(X, Y)
    X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y, n_classes)
    loss_mean = ((X_proj_means - A_proj_means)**2).mean()
    loss_var = ((X_proj_vars - A_proj_vars)**2).mean()
    return loss_mean + loss_var


loss_fn = loss_frechet

# ======= Optimize =======
lr = 0.1
steps = 400
optimizer = torch.optim.Adam([A, b], lr=lr)

deeputility.invert([X_B],
                   loss_fn,
                   optimizer,
                   steps=steps,
                   pre_fn=reconstruct,
                   #    track_history=True,
                   #    track_history_every=10,
                   plot=True,
                   )

# for x, step in zip(*zip(*history)):
#     utility.plot_stats(x, colors=['r'] * len(history))

# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:",
      dataset.cross_entropy(X_B_orig, Y_B).item())

plt.title("target data A")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="undistorted data B", alpha=0.4)
utility.plot_stats([X_A[Y_A == 0], X_B_proc[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B_proc[Y_B == 1]])
plt.legend()
plt.show()


# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
