"""Testing reconstruction methods"""
import os
import sys

import random

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt

import numpy as np

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import inversion
import datasets

from debug import debug

try:
    get_ipython()   # pylint: disable=undefined-variable
    interactive_notebook = True
except:
    interactive_notebook = False


if interactive_notebook:
    import importlib
    importlib.reload(utility)
    importlib.reload(inversion)
    importlib.reload(datasets)


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'MNIST'], required=True)
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_drive", action="store_true")
parser.add_argument("--use_var", action="store_true")
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_random_projections", type=int, default=256)
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset GMM'.split())
    # args.nn_steps = 500
    # args.inv_steps = 500
    # args.batch_size = -1

    # args = parser.parse_args('-dataset MNIST'.split())
    # args.inv_steps = 100
    # args.batch_size = 64
    # args.inv_lr = 0.01

    args = parser.parse_args('-dataset CIFAR10'.split())
    args.inv_steps = 600
    args.batch_size = 256

    args.n_random_projections = 1024
    args.use_drive = True
    # args.use_var = True
else:
    args = parser.parse_args()


print("#", __doc__)
print("# on", args.dataset)


# ======= Hyperparameters =======
print("Hyperparameters:")
print(utility.dict_to_str(vars(args), '\n'), '\n')

# ======= Set Seeds =======
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Neural Network
nn_lr = args.nn_lr
nn_steps = args.nn_steps
nn_resume_training = args.nn_resume_train
nn_reset_training = args.nn_reset_train

# Random Projections
n_random_projections = args.n_random_projections

# Inversion
inv_lr = args.inv_lr
inv_steps = args.inv_steps

# ======= Device =======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on '{DEVICE}'\n")

# ======= Create Dataset =======

if args.dataset == 'CIFAR10':
    dataset = datasets.CIFAR10(load_data=True)
elif args.dataset == 'MNIST':
    dataset = datasets.MNIST()

MODELDIR = dataset.data_dir

A, B, B_val = dataset.get_datasets()


def data_loader(D):
    return DataLoader(D, batch_size=64, shuffle=True)


DATA_A = data_loader(A)
# DATA_A = None

n_dims = dataset.n_dims
n_classes = dataset.n_classes

# ======= Neural Network =======
model_path, net = dataset.net()
net.to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=nn_lr)
utility.train(net, DATA_A, criterion, optimizer,
              model_path=model_path,
              epochs=nn_steps,
              resume_training=nn_resume_training,
              reset=nn_reset_training,
              plot=True,
              use_drive=args.use_drive,
              )


# ======= NN Project =======
net_layers = utility.get_bn_layers(net)
layer_activations = [None] * len(net_layers)


def layer_hook_wrapper(idx):
    def hook(_module, _inputs, outputs):
        layer_activations[idx] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN(data):
    inputs, labels = data
    outputs = net(inputs)
    return outputs


def project_NN_all(data):
    inputs, labels = data
    outputs = net(inputs)
    return [inputs] + layer_activations + [outputs]
    # XXXXXXXXXXXXXXXXX no outputs


# ======= Random Projections =======
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)


def identity(data): return data[0]


STD = not args.use_var

path = os.path.join(MODELDIR, "stats_inputs.pt")
path_cc = os.path.join(MODELDIR, "stats_inputs-CC.pt")
mean_A, std_A = utility.collect_stats(
    identity, DATA_A, n_classes, class_conditional=False,
    std=STD, path=path, device=DEVICE)
mean_A_C, std_A_C = utility.collect_stats(
    identity, DATA_A, n_classes, class_conditional=True,
    std=STD, path=path_cc, device=DEVICE)


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
relu_bias = (torch.randn((1, n_random_projections),
                         device=DEVICE) * std_A.max())
relu_bias_C = (torch.randn((n_classes, n_random_projections), device=DEVICE)
               * std_A_C.max(dim=1, keepdims=True)[0].reshape(n_classes, 1))


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


# ======= Loss Function =======
def regularization(x):
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    loss_var = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)
    return loss_var * 0.001


# debug.expand = False


# @debug
def loss_stats(m_a, s_a, m_b, s_b):
    if isinstance(m_a, list):
        assert len(m_a) == len(m_b) and len(s_a) == len(s_b), \
            "lists need to of same length"
        loss_mean = sum(((ma - mb)**2).mean()
                        for ma, mb in zip(m_a, m_b)) / len(m_a)
        loss_std = sum(((sa - sb)**2).mean()
                       for sa, sb in zip(s_a, s_b)) / len(m_a)
    else:
        loss_mean = ((m_a - m_b)**2).mean()
        loss_std = ((s_a - s_b)**2).mean()
    return loss_mean + loss_std


def loss_fn_wrapper(name, project, class_conditional):
    _name = name.replace(' ', '-')
    stats_path = os.path.join(MODELDIR, f"stats_{_name}.pt")
    m_a, s_a = utility.collect_stats(
        project, DATA_A, n_classes, class_conditional,
        std=STD, path=stats_path, device=DEVICE, use_drive=args.use_drive)

    # @debug
    def _loss_fn(data, m_a=m_a, s_a=s_a, project=project, class_conditional=class_conditional):
        inputs, labels = data
        outputs = project(data)
        last_layer = outputs[-1]
        m, s = utility.get_stats(
            outputs, labels, n_classes, class_conditional, std=STD)
        # loss = loss_stats(m_a, s_a, m, s)
        loss = (10 * loss_stats(m_a[1:-1], s_a[1:-1], m[1:-1], s[1:-1])
                + 0.001 * regularization(inputs)
                # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                + criterion(last_layer, labels)
                )
        return loss
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
    loss_fn_wrapper(
        name="NN ALL",
        project=project_NN_all,
        class_conditional=False,
    ),
    # loss_fn_wrapper(
    #     name="NN ALL CC",
    #     project=project_NN_all,
    #     class_conditional=True,
    # ),
    # loss_fn_wrapper(
    #     name="RP",
    #     project=project_RP,
    #     class_conditional=False,
    # ),
    # loss_fn_wrapper(
    #     name="RP CC",
    #     project=project_RP_CC,
    #     class_conditional=True,
    # ),
    # loss_fn_wrapper(
    #     name="RP ReLU",
    #     project=project_RP_relu,
    #     class_conditional=False,
    # ),
    # loss_fn_wrapper(
    #     name="RP ReLU CC",
    #     project=project_RP_relu_CC,
    #     class_conditional=True,
    # ),
    # loss_fn_wrapper(
    #     name="combined",
    #     project=combine(project_NN_all, project_RP_CC),
    #     class_conditional=True,
    # ),
]


def im_show(batch):
    with torch.no_grad():
        img_grid = torchvision.utils.make_grid(batch.cpu(), nrow=10)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()


# ======= Optimize =======
metrics = defaultdict(dict)


def jitter(x):
    off1, off2 = torch.randint(low=-2, high=2, size=(2, 1))
    x = torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    return x


def grad_norm_fn(x):
    return min(x, 10)  # torch.sqrt(x) if x > 1 else x
    # return np.sqrt(x) if x > 1 else x


for method, loss_fn in methods:
    print("\n## Method:", method)

    batch = torch.randn((args.batch_size, *dataset.input_shape),
                        device=DEVICE, requires_grad=True)
    labels = torch.LongTensor(range(args.batch_size)).to(DEVICE) % n_classes
    DATA = (batch, labels)

    print("Before:")
    im_show(batch)

    # def inputs_pre_fn(inputs):
    #     with torch.no_grad():
    #         inputs = perturb(inputs)
    #     outputs = preprocess(inputs)
    #     return outputs

    optimizer = torch.optim.Adam([batch], lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = inversion.inversion([DATA],
                               loss_fn,
                               optimizer,
                               #    scheduler=scheduler,
                               steps=inv_steps,
                               # steps=2,
                               # data_pre_fn=data_pre_fn,
                               inputs_pre_fn=jitter,
                               #    track_history=True,
                               #    track_history_every=10,
                               plot=True,
                               use_amp=args.use_amp,
                               grad_norm_fn=grad_norm_fn,
                               )

    # ======= Result =======
    print("Inverted:")
    im_show(batch)
