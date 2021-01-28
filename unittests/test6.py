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

from utils import utility
from utils import datasets
from utils import nets

if 'ipykernel_launcher' in sys.argv:
    import importlib
    importlib.reload(utility)
    importlib.reload(datasets)

print(__doc__)

cmaps = utility.categorical_colors(2)


# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
# Gaussian Mixture Model

n_classes = 3
dataset = datasets.MULTIGMM(
    n_dims=2,
    n_classes=n_classes,
    n_modes=8,
    scale_mean=5,
    scale_cov=2,
    mean_shift=20,
    n_samples_A=100,
    n_samples_B=100,
)

X_A, Y_A = dataset.A.tensors
X_B, Y_B = dataset.B.tensors
# means_A, _ = utility.c_stats(X_A, Y_A, n_classes)

# distorted Dataset B
distort_matrix = torch.eye(2) + 1 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


X_B_orig, Y_B = X_B, Y_B
X_B = distort(X_B_orig)


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
utility.train(net, [(X_A, Y_A)], criterion, optimizer,
              model_path=model_path,
              epochs=steps,
              #   resume=nn_resume_training,
              plot=True,
              )
net.eval()
# dataset.plot(net=net)
# plt.show()

print("Before:")
print("Cross Entropy of A:", dataset.cross_entropy(X_A).item())
print("Cross Entropy of B:", dataset.cross_entropy(X_B).item())


# ======= Collect Projected Stats from A =======
net_layers = utility.get_child_modules(net)[:-1]
layer_activations = [None] * len(net_layers)


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


# ======= reconstruct Model =======
A = torch.eye((2), requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def reconstruct(X):
    return X @ A + b


# ======= Loss Function =======
with torch.no_grad():
    X_A_proj = project(X_A)
A_proj_means, A_proj_vars = utility.c_stats(X_A_proj, Y_A, n_classes)


# def loss_frechet(X, Y=Y_B):
#     X_proj = project(X)
#     X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y, n_classes)
#     diff_mean = ((X_proj_means - A_proj_means)**2).sum(dim=0).mean()
#     diff_var = (X_proj_vars + A_proj_vars
#                 - 2 * (X_proj_vars * A_proj_vars).sqrt()
#                 ).sum(dim=0).mean()
#     loss = (diff_mean + diff_var)
#     return loss


def loss_fn(data):
    X, Y = data
    X = reconstruct(X)
    X_proj = project(X)

    X_proj_means, X_proj_vars = utility.c_stats(X_proj, Y, n_classes)
    loss_mean = (X_proj_means - A_proj_means).norm(dim=1).mean()
    loss_var = (X_proj_vars - A_proj_vars).norm(dim=1).mean()

    loss = loss_mean + loss_var

    info = {
        'loss': loss,
        '[losses] mean': loss_mean.item(),
        '[losses] var': loss_var.item(),
        'c-entropy': dataset.cross_entropy(X),
    }

    return info


# loss_fn = loss_frechet

# ======= Optimize =======
lr = 0.1
steps = 400
optimizer = torch.optim.Adam([A, b], lr=lr)
# scheduler = ReduceLROnPlateau(optimizer, verbose=True)

utility.invert([(X_B, Y_B)],
               loss_fn,
               optimizer,
               #    scheduler=scheduler,
               steps=steps,
               plot=True,
               track_grad_norm=True,
               )

# for x, step in zip(*zip(*history)):
#     utility.plot_stats(x, colors=['r'] * len(history))
# ======= Result =======
X_B_proc = reconstruct(X_B).detach()
print("After Reconstruction:")
print("Cross Entropy of B:", dataset.cross_entropy(X_B_proc).item())
print("Cross Entropy of undistorted B:",
      dataset.cross_entropy(X_B_orig).item())

plt.title("target data A")
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.scatter(X_B_orig[:, 0], X_B_orig[:, 1],
            c='orange', alpha=0.4, label="undistorted data B")
for c in range(n_classes):
    utility.plot_stats([X_A[Y_A == c], X_B_proc[Y_B == c]])
plt.legend()
plt.show()


# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
