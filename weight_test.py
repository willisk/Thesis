"""Testing reconstruction methods"""
import os
import sys

import random

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision

import matplotlib.pyplot as plt
# plt.style.use('default')

import numpy as np

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import inversion
import datasets
import debug
import nets

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
    importlib.reload(debug)
    importlib.reload(nets)

from debug import debug


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'GMM', 'MNIST'], required=True)
parser.add_argument("-seed", type=int, default=-1)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_std", action="store_true")
parser.add_argument("--use_jitter", action="store_true")
parser.add_argument("--plot_ideal", action="store_true")
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_random_projections", type=int, default=256)
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)
parser.add_argument("-f_reg", type=float, default=0.001)
parser.add_argument("-f_crit", type=float, default=1)
parser.add_argument("-f_stats", type=float, default=10)
parser.add_argument("-size_A", type=int, default=-1)
parser.add_argument("-size_B", type=int, default=64)
parser.add_argument("-distort_strength", type=float, default=0.03)

# GMM
parser.add_argument("-g_modes", type=int, default=3)
parser.add_argument("-g_scale_mean", type=float, default=2)
parser.add_argument("-g_scale_cov", type=float, default=20)
parser.add_argument("-g_mean_shift", type=float, default=0)

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset GMM'.split())
    # args.nn_steps = 500
    # args.inv_steps = 500
    # args.batch_size = -1

    args = parser.parse_args('-dataset MNIST'.split())
    # args.nn_steps = 5
    args.inv_steps = 2
    # args.batch_size = 64
    args.size_A = 64
    # # args.size_B = 10
    # # args.n_random_projections = 1024
    # args.inv_lr = 0.05
    # args.distort_strength = 0.5

    # args = parser.parse_args('-dataset CIFAR10'.split())
    # args.inv_steps = 1
    # args.batch_size = 64
    args.seed = 0

    args.size_B = 2
    args.plot_ideal = True
    # args.nn_resume_train = True
    # args.nn_reset_train = True
    # args.use_std = True
else:
    args = parser.parse_args()

USE_DRIVE = True

print("#", __doc__)
print("# on", args.dataset)


# ======= Hyperparameters =======
print("Hyperparameters:")
print(utility.dict_to_str(vars(args), '\n'), '\n')

# ======= Set Seeds =======


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


if args.seed != -1:
    set_seed(args.seed)

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


if args.dataset == 'GMM':
    dataset = datasets.MULTIGMM(
        input_shape=(20,),
        n_classes=3,
        n_modes=args.g_modes,
        scale_mean=args.g_scale_mean,
        scale_cov=args.g_scale_cov,
        mean_shift=args.g_mean_shift,
        n_samples_A=1000,
        n_samples_B=100,
        n_samples_B_val=100,
    )
elif args.dataset == 'CIFAR10':
    dataset = datasets.CIFAR10()
elif args.dataset == 'MNIST':
    dataset = datasets.MNIST()

MODELDIR = dataset.data_dir

A, B, B_val = dataset.get_datasets(size_A=args.size_A, size_B=args.size_B)


DATA_A = utility.DataL(
    A, batch_size=args.batch_size, shuffle=True, device=DEVICE)
DATA_B = utility.DataL(
    B, batch_size=args.batch_size, shuffle=True, device=DEVICE)
DATA_B_val = utility.DataL(
    B_val, batch_size=args.batch_size, shuffle=True, device=DEVICE)

input_shape = dataset.input_shape
n_dims = dataset.n_dims
n_classes = dataset.n_classes


STD = args.use_std
stats_path = os.path.join(MODELDIR, "stats_{}.pt")

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
              use_drive=USE_DRIVE,
              )
net.eval()


# ======= NN Project =======
net_layers = utility.get_child_modules(net)[:-1]
layer_activations = [None] * len(net_layers)
net_last_outputs = None


def layer_hook_wrapper(idx):
    def hook(_module, _inputs, outputs):
        layer_activations[idx] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN_all(data):
    global net_last_outputs
    inputs, labels = data
    net_last_outputs = net(inputs)
    return [inputs] + layer_activations


# ======= Loss Function =======
# @debug


def loss_stats(stats_a, stats_b):
    if not isinstance(stats_a, list):
        stats_a, stats_b = [stats_a], [stats_b]
    assert len(stats_a) == len(stats_b), "lists need to be of same length"
    loss_m = []
    loss_s = []
    for (ma, sa), (mb, sb) in zip(stats_a, stats_b):
        if ma.ndim == 1:
            loss_m.append((ma.squeeze() - mb.squeeze()).norm())
            loss_s.append((sa.squeeze() - sb.squeeze()).norm())
        else:   # class_conditional
            # one feature
            if np.prod(ma.shape) == ma.shape[0] or np.prod(mb.shape) == mb.shape[0]:
                loss_m.append((ma.squeeze() - mb.squeeze()).abs().mean())
                loss_s.append((sa.squeeze() - sb.squeeze()).abs().mean())
            else:
                loss_m.append((ma.squeeze() - mb.squeeze()).norm(dim=1).mean())
                loss_s.append((sa.squeeze() - sb.squeeze()).norm(dim=1).mean())
    return loss_m, loss_s


weights_m = torch.ones((len(net_layers) + 1,), requires_grad=True)
weights_s = torch.ones((len(net_layers) + 1,), requires_grad=True)


def softmax(Z):
    with torch.no_grad():
        s = torch.max(Z)
    A = torch.exp(Z - s)
    return A / A.sum()


def loss_fn_wrapper(name, project, class_conditional):
    _name = name.replace(' ', '-')

    stats_A = utility.collect_stats(
        DATA_A, project, n_classes, class_conditional,
        std=STD, path=stats_path.format(_name), device=DEVICE, use_drive=USE_DRIVE)

    def _loss_fn(data, project=project, class_conditional=class_conditional):
        global net_last_outputs
        net_last_outputs = None

        inputs, labels = data
        with torch.no_grad():
            outputs = project(data)

            stats = utility.get_stats(
                outputs, labels, n_classes, class_conditional=class_conditional, std=STD)

            loss_m, loss_s = loss_stats(stats, stats_A)
        loss = (torch.as_tensor(loss_m) * softmax(weights_m)).sum()
        loss += (torch.as_tensor(loss_s) * softmax(weights_s)).sum()
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
    #     name="NN ALL + RP CC",
    #     project=combine(project_NN_all, project_RP_CC),
    #     class_conditional=True,
    # ),
]


# ======= Optimize =======


def grad_norm_fn(x):
    return min(x, 10)  # torch.sqrt(x) if x > 1 else x


for method, loss_fn in methods:
    print("\n## Method:", method)

    # def callback_fn(epoch):
    #     if epoch % 10 == 0 and epoch > 0:
    #         print(f"\nepoch {epoch}:\
    #                 \tpsnr {utility.average_psnr(DATA_B, invert_fn)}", flush=True)
    #         im_show(invert_fn(show_batch))
    #         print(flush=True)

    optimizer = torch.optim.Adam([weights_m, weights_s], lr=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = inversion.invert(DATA_A,
                            loss_fn,
                            optimizer,
                            #    scheduler=scheduler,
                            steps=inv_steps,
                            # steps=2,
                            # data_pre_fn=data_pre_fn,
                            #    track_history=True,
                            #    track_history_every=10,
                            plot=True,
                            use_amp=args.use_amp,
                            # grad_norm_fn=grad_norm_fn,
                            # callback_fn=callback_fn,
                            track_per_batch=True,
                            )
