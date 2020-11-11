"""Testing reconstruction by matching
statistics on neural network feature
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
import statsnet
import deepinversion
import shared
import nets

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(deepinversion)
    importlib.reload(shared)

print(__doc__)

cmaps = utility.categorical_colors(2)


# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
# Gaussian Mixture Model

n_classes = 3

dataset = datasets.DatasetGMM(
    n_dims=2,
    n_classes=n_classes,
    n_modes=8,
    scale_mean=5,
    scale_cov=2,
    # mean_shift=20,
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


# ======= Neural Network =======
lr = 0.01
steps = 200
layer_dims = [2, 16, 16, 16, n_classes]
# layer_dims = [2, 4, 4, 4, n_classes]
model_path = os.path.join(
    PWD, f"models/net_GMM_{'-'.join(map(repr, layer_dims))}.pt")
net = nets.FCNet(layer_dims)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
utility.train(net, dataset.train_loader(), criterion, optimizer,
              model_path=model_path,
              num_epochs=steps,
              #   resume_training=True,
              plot=True,
              )
# dataset.plot(net=net)
# plt.show()

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A, Y_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B, Y_B).item())


# ======= Collect Projected Stats from A =======
net_layers = utility.net_get_relevant_layers(net, ignore_types=["activation"])
layer_activations = [0] * len(net_layers)


def layer_hook_wrapper(l):
    def hook(module, inputs, outputs):
        layer_activations[l] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))

# # skip last, skip relu
# net.main[-3].register_forward_hook(hook)


def project(X):
    net(X)
    return torch.cat(layer_activations, dim=1)


with torch.no_grad():
    X_A_proj = project(X_A)
A_proj_means, A_proj_vars, _ = utility.c_mean_var(X_A_proj, Y_A)

# ======= Preprocessing Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def preprocessing(X):
    return X @ A + b


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
# scheduler = ReduceLROnPlateau(optimizer, verbose=True)

history = deepinversion.deep_inversion(X_B,
                                       loss_fn,
                                       optimizer,
                                       #    scheduler=scheduler,
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
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc, Y_B).item())
plt.title("Data A")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="Data A", alpha=0.5)
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="preprocessed Data B", alpha=0.5)
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', label="unperturbed Data B", alpha=0.4)
for c in range(n_classes):
    utility.plot_stats([X_A[Y_A == c], X_B_proc[Y_B == c]])
plt.legend()
plt.show()


# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (preprocessing(perturb(Id)) - Id).norm(2).item()
