"""Testing reconstruction by matching
class-conditional statistics
"""
import os
import sys

import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import datasets
import deepinversion
import shared

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(deepinversion)
    importlib.reload(shared)

print(__doc__)
cmaps = utility.categorical_colors(2)

# Tensorboard
LOGDIR = os.path.join(PWD, "projectOK/runs")
shared.init_summary_writer(log_dir=LOGDIR)
# writer = shared.get_summary_writer("test4")


# ======= Set Seeds =======
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

# perturbed Dataset B
perturb_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)


def perturb(X):
    return X @ perturb_matrix + perturb_shift


X_B_orig, Y_B = dataset.sample(n_samples_per_class=100)
X_B = perturb(X_B_orig)

# Y_B_orig = Y_B
# Y_A.fill_(0)
# Y_B = Y_B * 0

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

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A, Y_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B).item())

# ======= Preprocessing Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def preprocessing(X):
    return X @ A + b


# ======= Collect Stats from A =======
# collect stats
# shape: [n_class, n_dims] = [2, 2]
A_means, A_vars, _ = utility.c_mean_var(X_A, Y_A, n_classes)


# ======= Loss Function =======
def loss_frechet(X, Y=Y_B):
    X_means, X_vars, _ = utility.c_mean_var(X, Y, n_classes)
    diff_mean = ((X_means - A_means)**2).sum(dim=0).mean()
    diff_var = (X_vars + A_vars - 2 * (X_vars * A_vars).sqrt()
                ).sum(dim=0).mean()
    loss = (diff_mean + diff_var)
    return loss


def loss_fn(X, Y=Y_B):
    # log likelihood * 2 - const:
    # diff_mean = (((X_means - A_means)**2 / X_vars.detach())).sum(dim=0)
    X_means, X_vars, _ = utility.c_mean_var(X, Y, n_classes)
    diff_mean = ((X_means - A_means)**2)
    diff_var = torch.abs(X_vars - A_vars).sqrt()
    loss = diff_mean.mean() + diff_var.mean()
    # loss = (0
    #         + diff_mean[0].mean()
    #         + diff_mean[1].mean()
    #         + diff_var[0].mean()
    #         + diff_var[1].mean()
    #         )
    return loss


# loss_fn = loss_frechet

# ======= Optimize =======
lr = 0.1
steps = 100
optimizer = torch.optim.Adam([A, b], lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, verbose=True)


deepinversion.deep_inversion([X_B],
                             loss_fn,
                             optimizer,
                             #    scheduler=scheduler,
                             steps=steps,
                             pre_fn=preprocessing,
                             #    track_history=True,
                             #    track_history_every=10,
                             plot=True,
                             )

# x_history, steps = zip(*history)
# for x in x_history[30:]:
#     utility.plot_stats(x[Y_B == 0], colors=['r'] * len(history))

# ======= Result =======
X_B_proc = preprocessing(X_B).detach()
print("After Pre-Processing:")
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc).item())
print("Cross Entropy of unperturbed B:",
      dataset.cross_entropy(X_B_orig, Y_B).item())

plt.title("Data A")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], alpha=0.4, label="Data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], alpha=0.4, label="preprocessed Data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', alpha=0.4, label="unperturbed Data B")
utility.plot_stats([X_A[Y_A == 0], X_B_proc[Y_B == 0]])
utility.plot_stats([X_A[Y_A == 1], X_B_proc[Y_B == 1]])
plt.legend()
plt.show()


# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (preprocessing(perturb(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")


# writer.close()
