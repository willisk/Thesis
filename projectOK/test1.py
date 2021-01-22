"""Simple regression test for recovering an affine transformation..
"""
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility

print('#', __doc__)

cmaps = utility.categorical_colors(2)
# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Create Dataset =======
X_A = torch.randn((3, 2)) * 4 + torch.randn((2)) * 10

distort_matrix = torch.eye(2) + 2 * torch.randn((2, 2))
distort_shift = 2 * torch.randn(2)


def distort(X):
    return X @ distort_matrix + distort_shift


# distorted Dataset B
X_B = distort(X_A)

plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B[:, 0], X_B[:, 1], c=cmaps[1], label="source data B")
plt.legend()
plt.show()

# ======= reconstruct Model =======
A = torch.eye(2, requires_grad=True)
b = torch.zeros((2), requires_grad=True)


def reconstruct(X):
    return X @ A + b


# ======= Loss Function =======
def loss_fn(X):
    X = reconstruct(X)
    return (X - X_A).norm()


# ======= Optimize =======
lr = 0.01
steps = 200
optimizer = torch.optim.Adam([A, b], lr=lr)

utility.invert([X_B],
               loss_fn,
               optimizer,
               steps=steps,
               plot=True,
               track_grad_norm=True,
               )

print("After Reconstruction:")
X_B_proc = reconstruct(X_B).detach()
plt.scatter(X_A[:, 0], X_A[:, 1], c=cmaps[0], label="target data A")
plt.scatter(X_B_proc[:, 0], X_B_proc[:, 1],
            c=cmaps[1], label="reconstructed data B")
plt.legend()
plt.show()

# L2 Reconstruction Error
Id = torch.eye(2)
l2_err = (reconstruct(distort(Id)) - Id).norm(2).item()
print(f"l2 reconstruction error: {l2_err:.3f}")
