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

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(PWD, "data")
MODELDIR = os.path.join(PWD, "models/CIFAR10")
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

from utility import debug, print_t

print("#", __doc__)


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument("-split_A", type=float, default=0.8)
parser.add_argument("-perturb_strength", type=float, default=1.5)
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-nn_width", type=int, default=16)
parser.add_argument("-nn_depth", type=int, default=4)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--nn_verifier", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("-n_random_projections", type=int, default=256)
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)
parser.add_argument("-seed", type=int, default=333)

if 'ipykernel_launcher' in sys.argv:
    args = parser.parse_args([])
    # args.nn_verifier = True
    args.nn_steps = 2
    args.inv_steps = 0
    args.batch_size = 32
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

# Random Projections
n_random_projections = args.n_random_projections
# Inversion
inv_lr = args.inv_lr
inv_steps = args.inv_steps

# ======= Device =======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on '{DEVICE}'")

# ======= Create Dataset =======
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataloader_params = {'batch_size': args.batch_size,
                     'shuffle': True}

CIF10 = CIFAR10(root=DATADIR, train=True,
                transform=img_transform, download=True)
B_val = CIFAR10(root=DATADIR, train=False,
                transform=img_transform, download=True)

n_A = int(len(CIF10) * split_A)
n_B = len(CIF10) - n_A

A, B = torch.utils.data.random_split(CIF10, (n_A, n_B))

DATA_A = DataLoader(A, **dataloader_params)
DATA_B = DataLoader(B, **dataloader_params)
DATA_B_val = DataLoader(B_val, **dataloader_params)


# ======= Perturbation =======
perturb_matrix = (torch.eye(n_dims) + perturb_strength *
                  torch.randn((n_dims, n_dims))).to(DEVICE)
perturb_shift = (perturb_strength * torch.randn(n_dims)).to(DEVICE)


def perturb(X):
    X_shape = X.shape
    X = X.reshape(-1, n_dims)
    out = X @ perturb_matrix + perturb_shift
    return out.reshape(X_shape)


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
model_path = os.path.join(MODELDIR, f"net_{model_name}.pt")
utility.train(net, DATA_A, criterion, optimizer,
              model_path=model_path,
              epochs=nn_steps,
              resume_training=nn_resume_training,
              reset=nn_reset_training,
              plot=True,
              use_drive=True,
              )
# accuracy_A = utility.net_accuracy(net, DATA_A)
accuracy_A = 0.99
print("USING DUMMY ACCURACY")
print(f"net accuracy: {accuracy_A * 100:.1f}%")

net_n_params = sum(p.numel() for p in net.parameters()
                   # if p.requires_grad
                   )
print(f"net parameters {net_n_params}")
print(f"net parameters {net_n_params / n_dims}")


if nn_verifier:
    verifier_path = os.path.join(MODELDIR, f"net_{model_name}_verifier.pt")
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
    outputs = layer_activations[-1]
    return outputs


def project_NN_all(data):
    inputs, labels = data
    net(inputs)
    outputs = [inputs] + layer_activations
    return outputs


# ======= Random Projections =======
# n_random_projections = net_n_params
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)


def identity(data): return data[0]


path = os.path.join(MODELDIR, "stats_inputs.pt")
path_cc = os.path.join(MODELDIR, "stats_inputs-CC.pt")
mean_A, std_A = utility.collect_stats(
    identity, DATA_A, n_classes, class_conditional=False,
    std=True, path=path, device=DEVICE)
mean_A_C_T, std_A_C = utility.collect_stats(
    identity, DATA_A, n_classes, class_conditional=True,
    std=True, path=path_cc, device=DEVICE)

mean_A = mean_A.reshape(-1, 1, 1)
mean_A_C = mean_A_C_T.T.reshape(n_classes, -1, 1, 1).contiguous()


# @debug
def project_RP(data):
    X, Y = data
    return (X - mean_A).reshape(-1, n_dims) @ RP


def project_RP_CC(data):
    X, Y = data
    X_proj_C = None
    for c in range(n_classes):
        X_proj_c = (X[Y == c] - mean_A_C[c]).reshape(-1, n_dims) @ RP
        if X_proj_C is None:
            X_proj_C = torch.empty((X.shape[0], n_random_projections),
                                   dtype=X_proj_c.dtype, device=X.device)
        X_proj_C[Y == c] = X_proj_c
    return X_proj_C


# Random ReLU Projections
relu_bias = (torch.randn((1, n_random_projections), device=DEVICE)
             * std_A.max())
relu_bias_C = (torch.randn((n_classes, n_random_projections), device=DEVICE)
               * std_A_C.max(dim=0, keepdims=True)[0].T)


def project_RP_relu(data):
    return F.relu(project_RP(data) + relu_bias)


def project_RP_relu_CC(data):
    X, Y = data
    return F.relu(project_RP_CC(data) + relu_bias_C[Y])

# ======= Combined =======


def combine(project1, project2):
    def _combined_fn(data):
        out1 = project1(data)
        out2 = project2(data)
        if not isinstance(out1, list):
            out1 = [out1]
        if not isinstance(out2, list):
            out2 = [out2]
        return out1 + out2
    return _combined_fn

# ======= Preprocessing Model =======


def preprocessing_model():
    M = torch.eye(n_dims, requires_grad=True, device=DEVICE)
    b = torch.zeros((n_dims), requires_grad=True, device=DEVICE)

    def preprocessing_fn(X):
        X_shape = X.shape
        X = X.reshape(-1, n_dims)
        return (X @ M + b).reshape(X_shape)

    return preprocessing_fn, (M, b)


# %%
# ======= Loss Function =======
def loss_stats(m_a, s_a, m_b, s_b):
    loss_mean = ((m_a - m_b)**2).mean()
    loss_std = ((s_a - s_b)**2).mean()
    return loss_mean + loss_std


from functools import wraps


importlib.reload(utility)


def loss_fn_wrapper(name, project, class_conditional):
    name = name.replace(' ', '-')
    stats_path = os.path.join(MODELDIR, f"stats_{model_name}_{name}.pt")
    m_a, s_a = utility.collect_stats(
        project, DATA_A, n_classes, class_conditional,
        std=True, path=stats_path, device=DEVICE)

    def _loss_fn(data, m_a=m_a, s_a=s_a, project=project, class_conditional=class_conditional):
        inputs, labels = data
        outputs = project(data)
        m, s = utility.get_stats(
            outputs, labels, n_classes, class_conditional, std=True)
        return loss_stats(m_a, s_a, m, s)
    return name, _loss_fn


methods = [
    # loss_fn_wrapper(
    #     name="NN",
    #     project=project_NN,
    #     class_conditional=False,
    # ),
    # loss_fn_wrapper(
    #     name="NN CC",
    #     project=project_NN,
    #     class_conditional=True,
    # ),
    # loss_fn_wrapper(
    #     name="NN ALL",
    #     project=project_NN_all,
    #     class_conditional=False,
    # ),
    loss_fn_wrapper(
        name="NN ALL CC",
        project=project_NN_all,
        class_conditional=True,
    ),
    loss_fn_wrapper(
        name="RP",
        project=project_RP,
        class_conditional=False,
    ),
    loss_fn_wrapper(
        name="RP CC",
        project=project_RP_CC,
        class_conditional=True,
    ),
    loss_fn_wrapper(
        name="RP ReLU",
        project=project_RP_relu,
        class_conditional=False,
    ),
    loss_fn_wrapper(
        name="RP ReLU CC",
        project=project_RP_relu_CC,
        class_conditional=True,
    ),
    loss_fn_wrapper(
        name="combined",
        project=combine(project_NN_all, project_RP_CC),
        class_conditional=True,
    ),
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


def to_device(X):
    return X.to(DEVICE)


for method, loss_fn in methods:
    print("## Method:", method)

    DATA_B.dataset.dataset.transform = img_transform

    preprocess, params = preprocessing_model()

    def pre_fn_x(inputs):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            inputs = perturb(inputs)
        outputs = preprocess(inputs)
        return outputs

    def pre_fn(data):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            inputs = perturb(inputs)
        outputs = preprocess(inputs)
        return (outputs, labels)

    optimizer = torch.optim.Adam(params, lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = deepinversion.deep_inversion(DATA_B,
                                        loss_fn,
                                        optimizer,
                                        #    scheduler=scheduler,
                                        steps=inv_steps,
                                        # steps=2,
                                        pre_fn=pre_fn,
                                        #    track_history=True,
                                        #    track_history_every=10,
                                        plot=True,
                                        use_amp=args.use_amp,
                                        )

    # ======= Result =======
    print("Results:")

    invert_transform = transforms.Compose([img_transform, pre_fn_x])
    DATA_B.dataset.dataset.transform = invert_transform
    DATA_B_val.dataset.transform = invert_transform

    # Loss
    # loss = accumulate_fn(DATA_B, loss_fn)
    loss = info['loss'][-1]
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

DATA_B.dataset.dataset.transform = img_transform
DATA_B_val.dataset.transform = img_transform

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

baseline['B (original)']['acc'] = accuracy_B
baseline['B (original)']['acc(val)'] = accuracy_B_val

baseline['B (perturbed)']['acc'] = accuracy_B_pert
baseline['B (perturbed)']['acc(val)'] = accuracy_B_val_pert

baseline['A']['acc'] = accuracy_A

if nn_verifier:
    baseline['B (perturbed)']['acc(ver)'] = accuracy_B_pert_ver
    baseline['B (original)']['acc(ver)'] = accuracy_B_ver
    baseline['A']['acc(ver)'] = accuracy_A_ver


print("\n# Summary")
print("=========\n")

utility.print_tabular(baseline, row_name="baseline")

print("\nReconstruction methods:")

utility.print_tabular(metrics, row_name="method")
