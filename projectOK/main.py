"""Testing reconstruction methods on high-dimensional Gaussian Mixtures
"""
import os
import sys

import argparse
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

print("#", __doc__)

cmaps = utility.categorical_colors(2)

# ======= Set Seeds =======
np.random.seed(9000)
torch.manual_seed(9000)

# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument("-n_classes", type=int, default=10)
parser.add_argument("-n_dims", type=int, default=20)
parser.add_argument("-n_samples", type=int, default=100)
parser.add_argument("-perturb_strength", type=float, default=1.5)
parser.add_argument("-g_modes", type=int, default=12)
parser.add_argument("-g_scale_mean", type=float, default=2)
parser.add_argument("-g_scale_cov", type=float, default=20)
parser.add_argument("-g_mean_shift", type=float, default=0)
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-nn_width", type=int, default=16)
parser.add_argument("-nn_depth", type=int, default=4)
parser.add_argument("-nn_depth", type=int, default=4)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)

if sys.argv[0] == 'ipykernel_launcher':
    args = parser.parse_args([])
else:
    args = parser.parse_args()

print("Hyperparameters:")
print(utility.dict_to_str(vars(args)), '\n')


# ======= Hyperparameters =======
# Dataset
n_classes = args.n_classes
n_dims = args.n_dims
perturb_strength = args.perturb_strength

# gmm
n_modes = args.g_modes
scale_mean = args.g_scale_mean
scale_cov = args.g_scale_cov
mean_shift = args.g_mean_shift
n_samples_per_class = args.n_samples

# Neural Network
nn_lr = args.nn_lr
nn_steps = args.nn_steps
nn_width = args.nn_width
nn_depth = args.nn_depth
nn_layer_dims = [n_dims] + [nn_width] * nn_depth + [n_classes]
nn_resume_training = args.nn_resume_train
nn_reset_training = args.nn_reset_train

# Random Projections
n_random_projections = args.n_random_projections

# Inversion
inv_lr = args.inv_lr
inv_steps = args.inv_steps

# ======= Create Dataset =======
# Gaussian Mixture Model
dataset = datasets.DatasetGMM(
    n_dims=n_dims,
    n_classes=n_classes,
    n_modes=n_modes,
    scale_mean=scale_mean,
    scale_cov=scale_cov,
    mean_shift=mean_shift,
    n_samples_per_class=n_samples_per_class,
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
              resume_training=nn_resume_training,
              reset=nn_reset_training,
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
    print("## Method:", method)

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
