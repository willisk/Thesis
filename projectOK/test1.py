""" Simple regression test for recovering an affine transformation
"""
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import deepinversion

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(deepinversion)


# ======= Set Seeds =======
np.random.seed(0)
torch.manual_seed(0)

# ======= Create Dataset =======
X_A = torch.randn((3, 2)) * 4 + torch.randn((2)) * 10

perturb_matrix = torch.eye(2) + 2 * torch.randn((2, 2))
perturb_shift = 2 * torch.randn(2)

X_B = X_A @ perturb_matrix + perturb_shift

plt.scatter(X_A[:, 0], X_A[:, 1], c='b', label="Data A")
plt.scatter(X_B[:, 0], X_B[:, 1], c='r', label="Data B")
plt.legend()
plt.show()

# ======= Preprocessing Model =======
A = torch.randn((2, 2), requires_grad=True)
b = torch.randn((2), requires_grad=True)


def preprocessing(X):
    return X @ A + b


# ======= Loss Function =======
criterion = torch.nn.MSELoss()


def loss_fn(X):
    return criterion(X, X_A)


optimizer = torch.optim.Adam([A, b], lr=0.01)

deepinversion.deep_inversion(X_B,
                             loss_fn,
                             optimizer,
                             steps=200,
                             pre_fn=preprocessing,
                             )

print("After Pre-Processing:")
X_B_proc = preprocessing(X_B).detach()
plt.scatter(X_A[:, 0], X_A[:, 1], c='b', label="Data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1], c='r', label="preprocessed Data B")
plt.legend()
plt.show()
