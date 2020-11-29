"""Testing reconstruction methods on high-dimensional Gaussian Mixtures
"""
import os
import sys

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import datasets
import statsnet
import deepinversion
import shared
import nets

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)

from utility import debug

print("#", __doc__)

# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument("-n_classes", type=int, default=10)
parser.add_argument("-n_dims", type=int, default=20)
parser.add_argument("-n_samples_A", type=int, default=500)
parser.add_argument("-n_samples_B", type=int, default=100)
parser.add_argument("-n_samples_valid", type=int, default=1000)
parser.add_argument("-perturb_strength", type=float, default=1.5)
parser.add_argument("-g_modes", type=int, default=12)
parser.add_argument("-g_scale_mean", type=float, default=2)
parser.add_argument("-g_scale_cov", type=float, default=20)
parser.add_argument("-g_mean_shift", type=float, default=0)
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-nn_width", type=int, default=16)
parser.add_argument("-nn_depth", type=int, default=4)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--nn_verifier", action="store_true")
parser.add_argument("-n_random_projections", type=int, default=16)
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)
parser.add_argument("-seed", type=int, default=333)

use_drive = True

cmaps = utility.categorical_colors(2)

if 'ipykernel_launcher' in sys.argv:
    args = parser.parse_args([])
    args.n_classes = 3
    args.g_modes = 3
    args.n_samples_A = 1000
    args.n_samples_B = 100
    args.n_samples_valid = 100
    args.nn_width = 16
    args.nn_reset = True
    args.nn_verifier = True
    use_drive = False
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
n_classes = args.n_classes
n_dims = args.n_dims
perturb_strength = args.perturb_strength

# gmm
n_modes = args.g_modes
scale_mean = args.g_scale_mean
scale_cov = args.g_scale_cov
mean_shift = args.g_mean_shift
n_samples_per_class_A = args.n_samples_A
n_samples_per_class_B = args.n_samples_B
n_samples_per_class_valid = args.n_samples_valid

# Neural Network
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
# Gaussian Mixture Model
dataset = datasets.DatasetGMM(
    n_dims=n_dims,
    n_classes=n_classes,
    n_modes=n_modes,
    scale_mean=scale_mean,
    scale_cov=scale_cov,
    mean_shift=mean_shift,
    n_samples_per_class=n_samples_per_class_A,
    device=DEVICE,
)

X_A, Y_A = dataset.X, dataset.Y
X_B, Y_B = dataset.sample(n_samples_per_class=n_samples_per_class_B)
X_B_val, Y_B_val = dataset.sample(
    n_samples_per_class=n_samples_per_class_valid)

perturb_matrix = (torch.eye(n_dims) + perturb_strength *
                  torch.randn((n_dims, n_dims))).to(DEVICE)
perturb_shift = (perturb_strength * torch.randn(n_dims)).to(DEVICE)


def perturb(X):
    return X @ perturb_matrix + perturb_shift


# perturbed Dataset B
X_B_orig = X_B
X_B = perturb(X_B_orig)
X_B_val = perturb(X_B_val)
DATA_A = (X_A.to(DEVICE), Y_A.to(DEVICE))

# ======= Neural Network =======
model_name = f"net_GMM_{'-'.join(map(repr, nn_layer_dims))}"
model_path = os.path.join(PWD, f"models/{model_name}.pt")
net = nets.FCNet(nn_layer_dims)
net.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=nn_lr)
utility.train(net, [DATA_A], criterion, optimizer,
              model_path=model_path,
              epochs=nn_steps,
              resume_training=nn_resume_training,
              reset=nn_reset_training,
              plot=True,
              use_drive=use_drive,
              )
utility.print_net_accuracy_batch(net, X_A, Y_A)

if nn_verifier:
    verifier_path = os.path.join(PWD, f"models/{model_name}_verifier.pt")
    verifier_net = nets.FCNet(nn_layer_dims)
    verifier_net.to(DEVICE)
    optimizer = torch.optim.Adam(verifier_net.parameters(), lr=nn_lr)
    utility.train(verifier_net, [DATA_A], criterion, optimizer,
                  model_path=verifier_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  )
    print("verifier ", end='')
    utility.print_net_accuracy_batch(verifier_net, X_A, Y_A)


# ======= NN Project =======
net_layers = utility.get_child_modules(net, ignore_types=["activation"])
layer_activations = [0] * len(net_layers)


def layer_hook_wrapper(l):
    def hook(module, inputs, outputs):
        layer_activations[l] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN(data):
    X, Y = data
    net(X)
    return layer_activations[-1]


def project_NN_all(data):
    X, Y = data
    net(X)
    return torch.cat(layer_activations, dim=1)


# ======= Random Projections =======
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)

mean_A, var_A = X_A.mean(dim=0), X_A.var(dim=0)
mean_A_C, var_A_C, _ = utility.c_mean_var(X_A, Y_A, n_classes)


def project_RP(data):
    X, Y = data
    return (X - mean_A) @ RP


def project_RP_CC(data):
    X, Y = data
    X_proj_C = torch.empty((X.shape[0], n_random_projections), device=X.device)
    for c in range(n_classes):
        X_proj_C[Y == c] = (X[Y == c] - mean_A_C[c]) @ RP
    return X_proj_C


# Random ReLU Projections
relu_bias = (torch.randn((1, n_random_projections), device=DEVICE)
             * var_A.max().sqrt())
relu_bias_C = (torch.randn((n_classes, n_random_projections), device=DEVICE)
               * var_A_C.max(dim=1)[0].sqrt().reshape(-1, 1))


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
    A = torch.eye(n_dims, requires_grad=True, device=DEVICE)
    b = torch.zeros((n_dims), requires_grad=True, device=DEVICE)

    def preprocessing_fn(X):
        return X @ A + b
    return preprocessing_fn, (A, b)


# ======= Loss Function =======
# def loss_frechet(X_proj_means, X_proj_vars, means_target, vars_target):
#     loss_mean = ((X_proj_means - means_target)**2).sum(dim=0).mean()
#     loss_var = (X_proj_vars + vars_target
#                 - 2 * (X_proj_vars * vars_target).sqrt()
#                 ).sum(dim=0).mean()
#     return loss_mean + loss_var


def loss_di(m, v, m_target, v_target):
    loss_mean = ((m - m_target)**2).mean()
    loss_var = ((v - v_target)**2).mean()
    return loss_mean + loss_var


def loss_frechet(X_proj_means, X_proj_vars, means_target, vars_target):
    loss_mean = ((X_proj_means - means_target)**2).sum(dim=0).mean()
    loss_var = (X_proj_vars + vars_target
                - 2 * (X_proj_vars * vars_target).sqrt()
                ).sum(dim=0).mean()
    return loss_mean + loss_var


def get_stats(inputs, labels, class_conditional):
    if class_conditional:
        mean, var, _ = utility.c_mean_var(inputs, labels, n_classes)
        return mean, var
    return inputs.mean(dim=0), inputs.var(dim=0)


def loss_fn_wrapper(loss_stats, project, class_conditional):
    with torch.no_grad():
        X_A_proj = project((X_A, Y_A))
        A_proj_means, A_proj_vars = get_stats(X_A_proj, Y_A, class_conditional)

    def _loss_fn(data, means_target=A_proj_means, vars_target=A_proj_vars, class_conditional=class_conditional):
        assert isinstance(data, tuple), f"data is not a tuple {data}"
        X, Y = data
        X_proj = project(data)
        X_proj_means, X_proj_vars = get_stats(X_proj, Y, class_conditional)
        return loss_stats(X_proj_means, X_proj_vars, means_target, vars_target)
    return _loss_fn


loss_stats = loss_frechet

methods = {
    "NN": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_NN,
        class_conditional=False,
    ),
    "NN CC": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_NN,
        class_conditional=True,
    ),
    "NN ALL": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_NN_all,
        class_conditional=False,
    ),
    "NN ALL CC": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_NN_all,
        class_conditional=True,
    ),
    "RP": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_RP,
        class_conditional=False,
    ),
    "RP CC": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_RP_CC,
        class_conditional=True,
    ),
    "RP ReLU": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_RP_relu,
        class_conditional=False,
    ),
    "RP ReLU CC": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=project_RP_relu_CC,
        class_conditional=True,
    ),
    "combined": loss_fn_wrapper(
        loss_stats=loss_stats,
        project=combine(project_NN_all, project_RP_CC),
        class_conditional=True,
    ),
}


# ======= Optimize =======
metrics = defaultdict(dict)

for method, loss_fn in methods.items():
    print("## Method:", method)

    preprocess, params = preprocessing_model()

    def pre_fn(data):
        X, Y = data
        with torch.no_grad():
            data = (preprocess(X), Y)
        return data

    optimizer = torch.optim.Adam(params, lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    DATA_B = (X_B.to(DEVICE), Y_B.to(DEVICE))
    deepinversion.deep_inversion([DATA_B],
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
    X_B_proc = preprocess(X_B).detach()
    X_B_val_proc = preprocess(X_B_val).detach()
    DATA_B_proc = (X_B_proc, Y_B)

    print("Results:")

    # Loss
    loss = loss_fn(DATA_B_proc).item()
    print(f"\tloss: {loss:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE)
    l2_err = (preprocess(perturb(Id)) - Id).norm(2).item() / Id.norm(2).item()
    print(f"\tl2 reconstruction error: {l2_err:.3f}")

    # Cross Entropy
    entropy = dataset.cross_entropy(X_B_proc, Y_B)
    print(f"\tcross entropy of B: {entropy:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy_batch(net, X_B_proc, Y_B)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")
    accuracy_val = utility.net_accuracy_batch(
        net, X_B_val_proc, Y_B_val)
    print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    if nn_verifier:
        accuracy_ver = utility.net_accuracy_batch(
            verifier_net, X_B_val_proc, Y_B_val)
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")

    metrics[method]['loss'] = loss
    metrics[method]['l2-err'] = l2_err
    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val
    if nn_verifier:
        metrics[method]['acc(ver)'] = accuracy_ver
    metrics[method]['c-entr'] = entropy


print("\n# Summary")
print("=========")

print("\nData A")
accuracy = utility.net_accuracy_batch(net, X_A, Y_A)
entropy = dataset.cross_entropy(X_A, Y_A).item()
print(f"cross entropy: {entropy:.3f}")
print(f"nn accuracy: {accuracy * 100:.1f} %")

print("\nperturbed Data B")
entropy = dataset.cross_entropy(X_B, Y_B).item()
print(f"cross entropy: {entropy:.3f}")
# accuracy = utility.net_accuracy_batch(net, X_B, Y_B)
utility.print_net_accuracy_batch(net, X_B, Y_B)
print("B valid ", end='')
utility.print_net_accuracy_batch(net, X_B_val, Y_B_val)
# print(f"nn accuracy: {accuracy * 100:.1f} %")
# print(f"nn accuracy B valid: {accuracy_val * 100:.1f} %")
if nn_verifier:
    # accuracy_ver = utility.net_accuracy_batch(verifier_net, X_B_val, Y_B_val)
    print("B valid verifier ", end='')
    utility.print_net_accuracy_batch(verifier_net, X_B_val, Y_B_val)
    # print(f"nn verifier accuracy: {accuracy_ver * 100:.1f} %")

utility.print_tabular(metrics, row_name="method")
