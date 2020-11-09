"""Testing reconstruction methods on
high-dimensional Gaussian Mixtures
"""
import os
import sys

from collections import defaultdict

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)

print(__doc__)

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(3)
torch.manual_seed(3)

# ======= Hyperparameters =======
# Dataset
n_classes = 3
n_dims = 2
perturb_strength = 1.5

# Neural Network
nn_lr = 0.01
nn_steps = 10
# nn_layer_dims = [n_dims, 16, 16, 16, n_classes]
nn_layer_dims = [n_dims, 4, 4, 4, n_classes]

# Random Projections
n_random_projections = 3

# Inversion
inv_lr = 0.1
inv_steps = 10


# ======= Create Dataset =======
# Gaussian Mixture Model
dataset = datasets.DatasetGMM(
    n_dims=n_dims,
    n_classes=n_classes,
    n_modes=2,
    scale_mean=5,
    scale_cov=2,
    # mean_shift=20,
    n_samples_per_class=100,
)

X_A, Y_A = dataset.X, dataset.Y
X_B_orig, Y_B = dataset.sample(n_samples_per_class=100)

perturb_matrix = torch.eye(n_dims) + perturb_strength * \
    torch.randn((n_dims, n_dims))
perturb_shift = perturb_strength * torch.randn(n_dims)


def perturbation(X):
    return X @ perturb_matrix + perturb_shift


# perturbed Dataset B
X_B = perturbation(X_B_orig)


# ======= Neural Network =======
model_path = os.path.join(
    PWD, f"models/net_GMM_{'-'.join(map(repr, nn_layer_dims))}.pt")
net = nets.FCNet(nn_layer_dims)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=nn_lr)
utility.train(net, dataset.train_loader(), criterion, optimizer,
              model_path=model_path,
              num_epochs=nn_steps,
              #   resume_training=True,
              plot=True,
              )
# dataset.plot(net=net)
# plt.show()
utility.print_net_accuracy(net, X_A, Y_A)


# ======= NN Project =======
feature_activation = None


def hook(module, inputs, outputs):
    global feature_activation
    feature_activation = outputs


# skip last, skip relu
net.main[-3].register_forward_hook(hook)


def project_NN(X):
    net(X)
    return feature_activation


# ======= Random Projections =======
RP = torch.randn((n_dims, n_random_projections))
RP = RP / RP.norm(2, dim=0)

# XXX: Change to cluster means
mean_A = X_A.mean(dim=0)


def project_RP(X):
    return (X - mean_A) @ RP


# ======= Preprocessing Model =======
def preprocessing_model():
    A = torch.eye(n_dims, requires_grad=True)
    b = torch.zeros((n_dims), requires_grad=True)

    def preprocessing(X):
        return X @ A + b
    return preprocessing, (A, b)


# ======= Loss Function =======
def loss_frechet(X_proj_means, X_proj_vars, means_target, vars_target):
    loss_mean = ((X_proj_means - means_target)**2).sum(dim=0).mean()
    loss_var = (X_proj_vars + vars_target
                - 2 * (X_proj_vars * vars_target).sqrt()
                ).sum(dim=0).mean()
    return loss_mean + loss_var


def loss_di(X_proj_means, X_proj_vars, means_target, vars_target):
    loss_mean = ((X_proj_means - means_target)**2).mean()
    loss_var = ((X_proj_vars - vars_target)**2).mean()
    return loss_mean + loss_var


def loss_fn_wrapper(loss_fn, project, class_conditional):
    with torch.no_grad():
        X_A_proj = project(X_A)
    if class_conditional:
        A_proj_means, A_proj_vars, _ = utility.c_mean_var(X_A_proj, Y_A)
    else:
        A_proj_means, A_proj_vars = X_A.mean(dim=0), X_A.var(dim=0)

    def _loss_fn(X, loss_fn=loss_fn, project=project, means_target=A_proj_means, vars_target=A_proj_vars, class_conditional=class_conditional):
        X_proj = project(X)
        if class_conditional:
            X_proj_means, X_proj_vars, _ = utility.c_mean_var(X_proj, Y_B)
        else:
            X_proj_means, X_proj_vars = X.mean(dim=0), X.var(dim=0)
        return loss_fn(X_proj_means, X_proj_vars, means_target, vars_target)
    return _loss_fn


methods = {
    "NN feature": loss_fn_wrapper(
        loss_frechet,
        project_NN,
        class_conditional=False,
    ),
    "NN feature CC": loss_fn_wrapper(
        loss_frechet,
        project_NN,
        class_conditional=True,
    ),
    "RP": loss_fn_wrapper(
        loss_frechet,
        project_RP,
        class_conditional=False,
    ),
    "RP CC": loss_fn_wrapper(
        loss_frechet,
        project_NN,
        class_conditional=True,
    ),
}

# ======= Optimize =======
metrics = defaultdict(dict)

for method, loss_fn in methods.items():
    print()
    print("# Method:", method)

    preprocessing, params = preprocessing_model()
    optimizer = torch.optim.Adam(params, lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    deepinversion.deep_inversion(X_B,
                                 loss_fn,
                                 optimizer,
                                 #    scheduler=scheduler,
                                 steps=inv_steps,
                                 pre_fn=preprocessing,
                                 #    track_history=True,
                                 #    track_history_every=10,
                                 plot=True,
                                 )

    # ======= Result =======
    X_B_proc = preprocessing(X_B).detach()

    print("Results:")

    # Loss
    loss = loss_fn(X_B_proc).item()
    print(f"\tloss: {loss:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims)
    l2_err = (preprocessing(perturbation(Id)) - Id).norm(2).item()
    print(f"\tl2 reconstruction error: {l2_err:.3f}")

    # Cross Entropy
    entropy = dataset.cross_entropy(X_B_proc, Y_B)
    print(f"\tcross entropy of B: {entropy:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, X_B_proc, Y_B)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")

    metrics[method]['loss'] = loss
    metrics[method]['l2 err'] = l2_err
    metrics[method]['accuracy'] = accuracy
    metrics[method]['cross-entropy'] = entropy


print()
print("Summary")
print("=======")

print()
print("Data A")
accuracy = utility.net_accuracy(net, X_A, Y_A)
entropy = dataset.cross_entropy(X_A, Y_A).item()
print(f"cross entropy: {entropy:.3f}")
print(f"nn accuracy: {accuracy * 100:.1f} %")

print()
print("perturbed Data B")
accuracy = utility.net_accuracy(net, X_B, Y_B)
entropy = dataset.cross_entropy(X_B, Y_B).item()
print(f"cross entropy: {entropy:.3f}")
print(f"nn accuracy: {accuracy * 100:.1f} %")

print()
utility.print_tabular(metrics, row_name="method")
