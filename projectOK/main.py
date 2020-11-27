"""Testing reconstruction methods on high-dimensional Gaussian Mixtures
"""
import os
import sys

import argparse
from collections import defaultdict
from copy import copy

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PWD, "data")
MODELDIR = os.path.join(PWD, "models")
sys.path.append(PWD)

import utility
import datasets
import statsnet
import deepinversion
import shared
import nets

print("#", __doc__)

cmaps = utility.categorical_colors(2)

# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument("-split_A", type=float, default=0.8)
parser.add_argument("-perturb_strength", type=float, default=1.5)
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-nn_width", type=int, default=16)
parser.add_argument("-nn_depth", type=int, default=4)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--nn_verifier", action="store_true")
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)
parser.add_argument("-seed", type=int, default=333)

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)
    args = parser.parse_args([])
    args.nn_width = 8
    args.nn_verifier = True
    args.nn_steps = 2
    args.inv_steps = 2
else:
    args = parser.parse_args()

print("Hyperparameters:")
print(utility.dict_to_str(vars(args), '\n'))
print()

# ======= Set Seeds =======
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ======= Hyperparameters =======
# Dataset
perturb_strength = args.perturb_strength
split_A = args.split_A


# Neural Network
model_name = "resnet34_CIF10"
n_dims, n_classes = 3 * 32 * 32, 10

nn_lr = args.nn_lr
nn_steps = args.nn_steps
nn_width = args.nn_width
nn_depth = args.nn_depth
nn_layer_dims = [n_dims] + [nn_width] * nn_depth + [n_classes]
nn_resume_training = args.nn_resume_train
nn_reset_training = args.nn_reset_train
nn_verifier = args.nn_verifier

# Inversion
inv_lr = args.inv_lr
inv_steps = args.inv_steps

# ======= Device =======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on '{DEVICE}'")

# ======= Perturbation =======
perturb_matrix = (torch.eye(n_dims) + perturb_strength *
                  torch.randn((n_dims, n_dims))).to(DEVICE)
perturb_shift = (perturb_strength * torch.randn(n_dims)).to(DEVICE)


def perturb(X):
    return X @ perturb_matrix + perturb_shift


# ======= Create Dataset =======
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_perturb = transforms.Compose([cifar_transform, perturb])
dataloader_params = {'batch_size': 64,
                     'shuffle': True}
CIF10 = torchvision.datasets.CIFAR10(root=DATADIR, train=True,
                                     download=True, transform=cifar_transform)
n_A = int(len(CIF10) * split_A)
A, B_orig = torch.utils.data.random_split(CIF10, (n_A, len(CIF10) - n_A))

B = copy(B_orig)
B.transform = transform_perturb
B_val = torchvision.datasets.CIFAR10(root=DATADIR, train=False,
                                     download=True, transform=cifar_transform)

A_loader = torch.utils.data.DataLoader(A, **dataloader_params)
B_loader = torch.utils.data.DataLoader(B, **dataloader_params)
B_val_loader = torch.utils.data.DataLoader(B_val, **dataloader_params)

# ======= Neural Network =======
from ext.cifar10pretrained.cifar10_models.resnet import resnet34 as ResNet34
from ext.cifar10pretrained.cifar10_download import main as download_resnet
download_resnet()
pretrained = True
net = ResNet34(pretrained=pretrained)
net.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=nn_lr)
model_path = os.path.join(MODELDIR, f"{model_name}.pt")
if not pretrained:
    utility.train(net, A_loader, criterion, optimizer,
                  model_path=model_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  plot=True,
                  use_drive=True,
                  )
print("Dataset A ", end='')
utility.print_net_accuracy(net, A_loader)

if nn_verifier:
    verifier_path = os.path.join(MODELDIR, f"{model_name}_verifier.pt")
    verifier_net = ResNet34()
    verifier_net.to(DEVICE)
    optimizer = torch.optim.Adam(verifier_net.parameters(), lr=nn_lr)
    utility.train(verifier_net, A_loader, criterion, optimizer,
                  model_path=verifier_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  )
    print("Dataset A verifier ", end='')
    utility.print_net_accuracy(verifier_net, A_loader)


# ======= NN Project =======
net_layers = utility.get_child_modules(
    net, ignore_types=["activation", "loss"])
layer_activations = [None] * len(net_layers)


def layer_hook_wrapper(l):
    def hook(module, inputs, outputs):
        layer_activations[l] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN(inputs, _labels):
    net(inputs)
    return layer_activations[-1]


def project_NN_all(inputs, _labels):
    net(inputs)
    return torch.cat(layer_activations, dim=1)


# ======= Preprocessing Model =======


def preprocessing_model():
    M = torch.eye(n_dims, requires_grad=True, device=DEVICE)
    b = torch.zeros((n_dims), requires_grad=True, device=DEVICE)

    def preprocessing_fn(X):
        return X @ M + b

    return preprocessing_fn, (M, b)


def loss_di(X_proj_means, X_proj_vars, means_target, vars_target):
    loss_mean = ((X_proj_means - means_target)**2).mean()
    loss_var = ((X_proj_vars - vars_target)**2).mean()
    return loss_mean + loss_var


def get_stats(inputs, labels, class_conditional):
    if class_conditional:
        mean, var, _ = utility.c_mean_var(inputs, labels, n_classes)
        return mean, var
    return inputs.mean(dim=0), inputs.var(dim=0)


def loss_fn_wrapper(loss_stats, project, class_conditional):
    m_target, v_target = utility.collect_stats(
        project, A_loader, n_classes, class_conditional)

    def _loss_fn(inputs, labels):
        X_proj = project(inputs, labels)
        m, v = get_stats(X_proj, labels, class_conditional)
        return loss_stats(m, v, m_target, v_target)
    return _loss_fn


loss_stats = loss_di

methods = {
    "NN": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_NN,
        class_conditional=False,
    ),
}

# ======= Optimize =======
metrics = defaultdict(dict)

for method, loss_fn in methods.items():
    print("## Method:", method)

    preprocess, params = preprocessing_model()
    optimizer = torch.optim.Adam(params, lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    deepinversion.deep_inversion(B_loader,
                                 loss_fn,
                                 optimizer,
                                 #    scheduler=scheduler,
                                 steps=inv_steps,
                                 pre_fn=preprocess,
                                 #    track_history=True,
                                 #    track_history_every=10,
                                 plot=True,
                                 )

    # ======= Result =======
    B_transform = B.transform
    B.transform = transforms.Compose([B.transform, preprocess])
    B_val.transform = B.transform

    print("Results:")

    # Loss
    total_count = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in B_loader:
            bs = len(inputs)
            total_loss = loss_fn(inputs, labels).item() * bs
            total_count += bs
    loss = total_loss / total_count
    print(f"\tloss: {loss:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE)
    l2_err = (preprocess(perturb(Id)) - Id).norm(2).item() / Id.norm(2).item()
    print(f"\trel. l2 reconstruction error: {l2_err:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, B_loader)
    accuracy_val = utility.net_accuracy(net, B_val_loader)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")
    print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    metrics[method]['loss'] = loss
    metrics[method]['l2-err'] = l2_err
    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val

    if nn_verifier:
        accuracy_ver = utility.net_accuracy(verifier_net, B_val_loader)
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")
        metrics[method]['acc(ver)'] = accuracy_ver

    B.transform = B_transform
    B_val.transform = B_transform


print("\n# Summary")
print("=========")

accuracy = utility.net_accuracy(net, A_loader)
print("\nData A")
print(f"nn accuracy: {accuracy * 100:.1f} %")

accuracy = utility.net_accuracy(net, B_loader)
accuracy_val = utility.net_accuracy(net, B_val_loader)
print("\nperturbed Data B")
print(f"nn accuracy: {accuracy * 100:.1f} %")
print(f"nn accuracy B valid: {accuracy_val * 100:.1f} %")
if nn_verifier:
    accuracy_ver = utility.net_accuracy(verifier_net, B_val_loader)
    print(f"nn verifier accuracy: {accuracy_ver * 100:.1f} %")

print()
utility.print_tabular(metrics, row_name="method")
