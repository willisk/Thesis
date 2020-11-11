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

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(8)

# ======= Create Dataset =======
# Gaussian Mixture Model

n_classes = 3

dataset = datasets.DatasetGMM(
    n_dims=2,
    n_classes=n_classes,
    n_modes=2,
    scale_mean=20,
    scale_cov=20,
    # mean_shift=20,
    n_samples_per_class=100,
)

dataset.Y = dataset.Y[:220]
X_A, Y_A = dataset.X, dataset.Y
# X_B, Y_B = dataset.sample(n_samples_per_class=1)

# print("likelihood of A:", torch.prod(dataset.logpdf(X_A, Y_A)).item())
dataset.pairwise_JS()
# print("likelihood of A:", dataset.logpdf(
#     dataset.gmms[0].means.repeat(3, 1), [0, 0, 1]))
# print("Cross Entropy of B:", dataset.logpdf(X_B, Y_B))
# print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B))


# print(dataset.gmms[1].means)
# print(dataset.gmms[1].weights)
for c in range(n_classes):
    def loss_fn(X):
        return dataset.pdf(X)
        # return dataset.pdf(X, torch.Tensor([c] * len(X)))
    dataset.plot(loss_fn=loss_fn, scatter=False, legend=True,)
    plt.show()
# dataset.plot(
#     loss_fn=loss_fn,
#     # scatter=False,
#     legend=True,
# )
# plt.legend()
# utility.plot_contourf_data(
#     X_B, loss_fn, n_grid=100, scale_grid=scale_grid,
#     cmap=cmap, levels=levels, contour=True, colorbar=True)
# plt.scatter(X_B[:, 0], X_B[:, 1], c='k')
plt.show()
