"""Testing reconstruction methods on high-dimensional Gaussian Mixtures
"""
import os
import sys

import argparse
from collections import defaultdict
from copy import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)

from utility import debug

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

if 'ipykernel_launcher' in sys.argv:
    args = parser.parse_args([])
    # args.nn_verifier = True
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
model_name = "resnet34"
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
# def to_device(X):
#     return X.to(DEVICE)

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, device='cpu', **kwargs):
        super(CIFAR10, self).__init__(*args, **kwargs)
        self.data = transforms.ToTensor()(self.data)
        self.data = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(self.data)
        self.data = self.data.to(device)
        self.targets = self.targets.to(device)


dataloader_params = {'batch_size': 64,
                     'shuffle': True}
CIF10 = CIFAR10(root=DATADIR, device=DEVICE, train=True, download=True)
B_val = CIFAR10(root=DATADIR, device=DEVICE, train=False, download=True)

n_A = int(len(CIF10) * split_A)
n_B = len(CIF10) - n_A

A, B = torch.utils.data.random_split(CIF10, (n_A, n_B))

DATA_A = DataLoader(A, **dataloader_params)
DATA_B = DataLoader(B, **dataloader_params)
DATA_B_val = DataLoader(B_val, **dataloader_params)

# ======= Neural Network =======
from ext.cifar10pretrained.cifar10_models.resnet import resnet34 as ResNet34
from ext.cifar10pretrained.cifar10_download import main as download_resnet
download = False
if download:
    download_resnet()
net = ResNet34(pretrained=download)
net.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=nn_lr)
model_path = os.path.join(MODELDIR, f"{model_name}.pt")
if not download:
    utility.train(net, DATA_A, criterion, optimizer,
                  model_path=model_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  plot=True,
                  use_drive=True,
                  )
accuracy_A = utility.net_accuracy(net, DATA_A)
print(f"net accuracy: {accuracy_A * 100:.1f}%")

net_n_params = sum(p.numel() for p in net.parameters()
                   # if p.requires_grad
                   )
print(f"net parameters {net_n_params}")

if nn_verifier:
    verifier_path = os.path.join(MODELDIR, f"{model_name}_verifier.pt")
    verifier_net = ResNet34()
    verifier_net.to(DEVICE)
    optimizer = torch.optim.Adam(verifier_net.parameters(), lr=nn_lr)
    utility.train(verifier_net, DATA_A, criterion, optimizer,
                  model_path=verifier_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  use_drive=True,
                  )
    accuracy_A_ver = utility.net_accuracy(verifier_net, DATA_A)
    print(f"verifier net accuracy: {accuracy_A_ver * 100:.1f}%")


# ======= NN Project =======
net_layers = utility.get_child_modules(net)
layer_activations = [None] * len(net_layers)


def layer_hook_wrapper(l):
    def hook(module, inputs, outputs):
        layer_activations[l] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN(data):
    inputs, labels = data
    net(inputs)
    return layer_activations[-1]


def project_NN_all(data):
    inputs, labels = data
    net(inputs)
    return torch.cat([inputs] + layer_activations, dim=1)


# ======= Random Projections =======
n_random_projections = net_n_params
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)

mean_A, std_A = X_A.mean(dim=0), X_A.std(dim=0)
mean_A_C, std_A_C = utility.c_mean_std(X_A, Y_A, n_classes)


def project_RP(data):
    X, Y = data
    return (X - mean_A) @ RP


def project_RP_CC(data):
    X, Y = data
    X_proj_C = None
    for c in range(n_classes):
        X_proj_c = (X[Y == c] - mean_A_C[c]) @ RP
        if X_proj_C is None:
            X_proj_C = torch.empty((X.shape[0], n_random_projections),
                                   dtype=X_proj_c.dtype, device=X.device)
        X_proj_C[Y == c] = X_proj_c
    return X_proj_C


# Random ReLU Projections
relu_bias = (torch.randn((1, n_random_projections), device=DEVICE)
             * std_A.max())
relu_bias_C = (torch.randn((n_classes, n_random_projections), device=DEVICE)
               * std_A_C.max(dim=1)[0].reshape(-1, 1))


def project_RP_relu(data):
    return F.relu(project_RP(data) + relu_bias)


def project_RP_relu_CC(data):
    X, Y = data
    return F.relu(project_RP_CC(data) + relu_bias_C[Y])

# ======= Combined =======


def combine(project1, project2):
    def _combined_fn(data):
        return torch.cat((project1(data), project2(data)), dim=1)
    return _combined_fn

# ======= Preprocessing Model =======


def preprocessing_model():
    M = torch.eye(n_dims, requires_grad=True, device=DEVICE)
    b = torch.zeros((n_dims), requires_grad=True, device=DEVICE)

    def preprocessing_fn(X):
        X = X.reshape(-1, 3 * 32 * 32)
        return (X @ M + b).reshape(-1, 3, 32, 32)

    return preprocessing_fn, (M, b)


# ======= Loss Function =======
def loss_stats(m_a, s_a, m_b, s_b):
    loss_mean = ((m_a - m_b)**2).mean()
    loss_std = ((s_a - s_b)**2).mean()
    return loss_mean + loss_std


def get_stats(inputs, labels, class_conditional):
    if class_conditional:
        mean, std = utility.c_mean_std(inputs, labels, n_classes)
        return mean, std
    return inputs.mean(dim=0), inputs.std(dim=0)


def loss_fn_wrapper(name, project, class_conditional):
    stats_path = os.path.join(MODELDIR, f"stats_{model_name}_{name}.pt")
    m_a, s_a = utility.collect_stats(
        project, DATA_A, n_classes, class_conditional, 
        std=True, path=stats_path, device=DEVICE)

    def _loss_fn(data, m_a=m_a, s_a=s_a, project=project, class_conditional=class_conditional):
        assert isinstance(data, tuple), f"data is not a tuple {data}"
        inputs, labels = data
        X_proj = project(data)
        m_b, s_b = get_stats(X_proj, labels, class_conditional)
        return loss_stats(m_a, s_a, m_b, s_b)
    return name, _loss_fn


methods = [
    # "NN": loss_fn_wrapper(
    #     project=project_NN,
    #     class_conditional=False,
    # ),
    # "NN CC": loss_fn_wrapper(
    #     project=project_NN,
    #     class_conditional=True,
    # ),
    loss_fn_wrapper(
        name="NN ALL",
        project=project_NN_all,
        class_conditional=False,
    ),
    # "NN ALL CC": loss_fn_wrapper(
    #     project=project_NN_all,
    #     class_conditional=True,
    # ),
    # "RP": loss_fn_wrapper(
    #     project=project_RP,
    #     class_conditional=False,
    # ),
    # "RP CC": loss_fn_wrapper(
    #     project=project_RP_CC,
    #     class_conditional=True,
    # ),
    # "RP ReLU": loss_fn_wrapper(
    #     project=project_RP_relu,
    #     class_conditional=False,
    # ),
    # "RP ReLU CC": loss_fn_wrapper(
    #     project=project_RP_relu_CC,
    #     class_conditional=True,
    # ),
    # "combined": loss_fn_wrapper(
    #     project=combine(project_NN_all, project_RP_CC),
    #     class_conditional=True,
    # ),
]


def accumulate_fn(data_loader, func):
    total = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            bs = len(inputs)
            total += func(inputs, labels).item() * bs
    return total / len(data_loader)


# ======= Optimize =======
metrics = defaultdict(dict)

for method, loss_fn in methods:
    print("## Method:", method)

    B.transform = perturb
    preprocess, params = preprocessing_model()

    def pre_fn(data):
        X, Y = data
        data = (preprocess(X), Y)
        return data

    optimizer = torch.optim.Adam(params, lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    deepinversion.deep_inversion(DATA_B,
                                 loss_fn,
                                 optimizer,
                                 #    scheduler=scheduler,
                                 steps=inv_steps,
                                 pre_fn=pre_fn,
                                 #    track_history=True,
                                 #    track_history_every=10,
                                 plot=True,
                                 )

    # ======= Result =======
    print("Results:")
    invert_transform = transforms.Compose([perturb, preprocess])
    B.transform = invert_transform
    B_val.transform = invert_transform

    # Loss
    loss = accumulate_fn(DATA_B, loss_fn)
    print(f"\tloss: {loss:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE)
    l2_err = (preprocess(perturb(Id)) - Id).norm(2).item() / Id.norm(2).item()
    print(f"\trel. l2 reconstruction error: {l2_err:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, DATA_B)
    accuracy_val = utility.net_accuracy(net, DATA_B_val)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")
    print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val

    if nn_verifier:
        accuracy_ver = utility.net_accuracy(verifier_net, DATA_B)
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")
        metrics[method]['acc(ver)'] = accuracy_ver
    metrics[method]['l2-err'] = l2_err
    metrics[method]['loss'] = loss

baseline = defaultdict(dict)

B.transform = None
B_val.transform = None

accuracy_B = utility.net_accuracy(net, DATA_B)
accuracy_B_val = utility.net_accuracy(net, DATA_B_val)
if nn_verifier:
    accuracy_B_ver = utility.net_accuracy(verifier_net, DATA_B)

B.transform = perturb
B_val.transform = perturb

accuracy_B_pert = utility.net_accuracy(net, DATA_B)
accuracy_B_val_pert = utility.net_accuracy(net, DATA_B_val)
if nn_verifier:
    accuracy_B_pert_ver = utility.net_accuracy(verifier_net, DATA_B)

baseline['A']['acc'] = accuracy_A
baseline['A']['acc(val)'] = float('NaN')

baseline['B (original)']['acc'] = accuracy_B
baseline['B (original)']['acc(val)'] = accuracy_B_val

baseline['B (perturbed)']['acc'] = accuracy_B_pert
baseline['B (perturbed)']['acc(val)'] = accuracy_B_val_pert

if nn_verifier:
    baseline['A']['acc(ver)'] = accuracy_A_ver
    baseline['B (perturbed)']['acc(ver)'] = accuracy_B_pert_ver
    baseline['B (original)']['acc(ver)'] = accuracy_B_ver

print("\n# Summary")
print("=========\n")

utility.print_tabular(baseline, row_name="baseline")

print("\nReconstruction methods:")

utility.print_tabular(metrics, row_name="method")
