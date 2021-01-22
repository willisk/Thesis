"""Testing reconstruction methods"""
import os
import sys

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
    importlib.reload(methods)
    importlib.reload(datasets)
    importlib.reload(debug)
    importlib.reload(nets)

from debug import debug


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'MNIST'], required=True)
parser.add_argument("-seed", type=int, default=0)
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
parser.add_argument("-show_after", type=int, default=50)
parser.add_argument("-r_distort_level", type=float, default=0.3)
parser.add_argument("-r_block_depth", type=int, default=4)
parser.add_argument("-r_block_width", type=int, default=4)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_std", action="store_true")
parser.add_argument("--use_jitter", action="store_true")
parser.add_argument("--plot_ideal", action="store_true")
parser.add_argument("--scale_each", action="store_true")
parser.add_argument("--reset_stats", action="store_true")

# # GMM
# parser.add_argument("-g_modes", type=int, default=3)
# parser.add_argument("-g_scale_mean", type=float, default=2)
# parser.add_argument("-g_scale_cov", type=float, default=20)
# parser.add_argument("-g_mean_shift", type=float, default=0)

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset CIFAR10'.split())
    args = parser.parse_args('-dataset MNIST'.split())
    # args.nn_steps = 5
    args.inv_steps = 3
    args.r_distort_level = 0.1
    # args.batch_size = 64
    # # args.size_B = 10
    args.n_random_projections = 512
    args.inv_lr = 0.01
    args.f_stats = 0.001
    # args.distort_level = 0.5

    args.seed = 0

    args.size_B = 64
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
utility.seed_everything(args.seed)

# ======= Device =======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on '{DEVICE}'\n")

# ======= Create Dataset =======

if args.dataset == 'CIFAR10':
    dataset = datasets.CIFAR10()
elif args.dataset == 'MNIST':
    dataset = datasets.MNIST()

A, B, C = dataset.get_datasets(size_A=args.size_A, size_B=args.size_B)


DATA_A = utility.DataL(
    A, batch_size=args.batch_size, shuffle=True, device=DEVICE)
DATA_B = utility.DataL(
    B, batch_size=args.batch_size, shuffle=True, device=DEVICE)
DATA_C = utility.DataL(
    C, batch_size=args.batch_size, shuffle=True, device=DEVICE)

input_shape = dataset.input_shape
n_dims = dataset.n_dims
n_classes = dataset.n_classes

# ======= Neural Network =======
nn_lr = args.nn_lr
nn_steps = args.nn_steps
nn_resume_training = args.nn_resume_train
nn_reset_training = args.nn_reset_train

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

if not 'ipykernel_launcher' in sys.argv[0]:
    utility.print_net_accuracy(net, DATA_A, estimate_epochs=10)

verifier_path, verifier_net = dataset.verifier_net()
if verifier_net:
    verifier_net.to(DEVICE)
    optimizer = torch.optim.Adam(verifier_net.parameters(), lr=nn_lr)
    utility.train(verifier_net, DATA_A, criterion, optimizer,
                  model_path=verifier_path,
                  epochs=nn_steps,
                  resume_training=nn_resume_training,
                  reset=nn_reset_training,
                  use_drive=USE_DRIVE,
                  )
    if not 'ipykernel_launcher' in sys.argv[0]:
        print("verifier ", end='')
        utility.print_net_accuracy(verifier_net, DATA_A, estimate_epochs=10)
print()

# ======= Distortion =======


class DistortionModel(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 3
        nch = input_shape[0]
        lambd = args.r_distort_level

        self.conv1 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.noise = nn.Parameter(
            torch.randn(input_shape).unsqueeze(0))

        self.conv1.weight.data.normal_()
        self.conv2.weight.data.normal_()
        self.conv1.weight.data *= lambd
        self.conv2.weight.data *= lambd
        for f in range(nch):
            self.conv1.weight.data[f][f][1][1] += 1
            self.conv2.weight.data[f][f][1][1] += 1

        self.conv1.bias.data.normal_()
        self.conv2.bias.data.normal_()
        self.conv1.bias.data *= lambd
        self.conv2.bias.data *= lambd

        self.noise.data *= lambd

    @torch.no_grad()
    def forward(self, inputs):
        outputs = inputs
        outputs = outputs + self.noise
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


distort = DistortionModel()
distort.eval()
distort.to(DEVICE)

# ======= Reconstruction Model =======


def conv1x1Id(n_chan):
    conv = nn.Conv2d(n_chan, n_chan,
                     kernel_size=1,
                     bias=False,
                     )
    conv.weight.data.fill_(0)
    for i in range(n_chan):
        conv.weight.data[i, i, 0, 0] = 1
    return conv


class ReconstructionModel(nn.Module):
    def __init__(self, relu_out=False, bias=True):
        super().__init__()

        utility.seed_everything(args.seed)

        n_chan = input_shape[0]
        self.conv1x1 = conv1x1Id(n_chan)
        self.bn = nn.BatchNorm2d(n_chan)

        self.invert_block = nn.Sequential(*[
            nets.InvertBlock(
                n_chan,
                args.r_block_width,
                noise_level=1 / np.sqrt(n + 1),
                relu_out=n < args.r_block_depth - 1,
                bias=bias,
            ) for n in range(args.r_block_depth)
        ])

    def forward(self, inputs):
        outputs = self.conv1x1(inputs)
        outputs = self.invert_block(outputs)
        outputs = self.bn(outputs)
        return outputs


@torch.no_grad()
def im_show(im_batch):
    s = 1.6
    img_grid = torchvision.utils.make_grid(
        im_batch.cpu(), nrow=10, normalize=True, scale_each=args.scale_each)
    plt.figure(figsize=(s * 10, s * len(im_batch)))
    plt.axis('off')
    plt.grid(b=None)
    plt.imshow(img_grid.permute(1, 2, 0))
    plt.show()
    print(flush=True)


show_batch = next(iter(DATA_B))[0][:50].to(DEVICE)

print("\nground truth:", flush=True)
im_show(show_batch[:10])

print("\ndistorted:")
im_show(distort(show_batch[:10]))

# ======= Optimize =======
inv_lr = args.inv_lr
inv_steps = args.inv_steps

metrics = defaultdict(dict)
plots = {}


def jitter(inputs):
    shifts = tuple(torch.randint(low=-2, high=2, size=(2,)))
    return torch.roll(inputs, shifts=shifts, dims=(2, 3))


def grad_norm_fn(x):
    return min(x, 10)  # torch.sqrt(x) if x > 1 else x


for method, loss_fn in methods.get_methods(DATA_A, net, dataset, args, DEVICE):
    print("\n\n\n## Method:", method)

    reconstruct = ReconstructionModel()
    reconstruct.train()
    reconstruct.to(DEVICE)

    def invert_fn(inputs):
        return reconstruct(distort(inputs))

    def data_loss_fn(data):
        inputs, labels = data
        if args.use_jitter:
            inputs = jitter(inputs)
        data_inv = (invert_fn(inputs), labels)
        info = loss_fn(data_inv)
        info[':mean: psnr'] = utility.average_psnr([data], invert_fn)
        if args.plot_ideal:
            with torch.no_grad():
                info['ideal'] = loss_fn(data)['loss'].item()
        return info

    def callback_fn(epoch, metrics):
        if epoch % args.show_after == 0:
            print(f"\nepoch {epoch}:", flush=True)
            im_show(invert_fn(show_batch[:10]))

    optimizer = torch.optim.Adam(reconstruct.parameters(), lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = utility.invert(DATA_B,
                          data_loss_fn,
                          optimizer,
                          #    scheduler=scheduler,
                          steps=inv_steps,
                          plot=True,
                          use_amp=args.use_amp,
                          #    grad_norm_fn=grad_norm_fn,
                          callback_fn=callback_fn,
                          track_grad_norm=True,
                          # track_per_batch=True,
                          )
    plots[method] = info

    # ======= Result =======
    reconstruct.eval()

    print("Inverted:")
    if len(show_batch) != len(B):
        print(f"{len(show_batch)} / {len(B)} ")
    im_show(invert_fn(show_batch))

    print("Results:")

    # Loss
    loss = info['loss'][-1]
    print(f"\tloss: {loss:.3f}")

    # PSNR
    psnr = utility.average_psnr(DATA_B, invert_fn)
    psnr_distort = utility.average_psnr(DATA_B, distort)
    print(f"\taverage PSNR: {psnr:.3f} | (distorted: {psnr_distort:.3f})")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE).reshape(-1, *input_shape)
    l2_err = (invert_fn(Id) - Id).norm().item() / Id.norm().item()
    l2_err_distort = (distort(Id) - Id).norm().item() / Id.norm().item()
    print(
        f"\trel. l2 reconstruction error: {l2_err:.3f} | (distorted: {l2_err_distort:.3f})")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, DATA_B, inputs_pre_fn=invert_fn)
    accuracy_val = utility.net_accuracy(
        net, DATA_C, inputs_pre_fn=invert_fn)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")

    print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val

    if verifier_net:
        accuracy_ver = utility.net_accuracy(
            verifier_net, DATA_B, inputs_pre_fn=invert_fn)
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")
        metrics[method]['acc(ver)'] = accuracy_ver
    metrics[method]['av. PSNR'] = psnr
    metrics[method]['l2-err'] = l2_err
    # metrics[method]['loss'] = loss

baseline = defaultdict(dict)


accuracy_A = utility.net_accuracy(net, DATA_A)
accuracy_B = utility.net_accuracy(net, DATA_B)
accuracy_C = utility.net_accuracy(
    net, DATA_C)

accuracy_B_pert = utility.net_accuracy(
    net, DATA_B, inputs_pre_fn=distort)
accuracy_C_pert = utility.net_accuracy(
    net, DATA_C, inputs_pre_fn=distort)

if verifier_net:
    accuracy_A_ver = utility.net_accuracy(
        verifier_net, DATA_A)
    accuracy_B_ver = utility.net_accuracy(
        verifier_net, DATA_B)
    accuracy_B_pert_ver = utility.net_accuracy(
        verifier_net, DATA_B, inputs_pre_fn=distort)

baseline['B (original)']['acc'] = accuracy_B
baseline['B (original)']['acc(val)'] = accuracy_C

baseline['B (distorted)']['acc'] = accuracy_B_pert
baseline['B (distorted)']['acc(val)'] = accuracy_C_pert

baseline['A']['acc'] = accuracy_A

if verifier_net:
    baseline['B (distorted)']['acc(ver)'] = accuracy_B_pert_ver
    baseline['B (original)']['acc(ver)'] = accuracy_B_ver
    baseline['A']['acc(ver)'] = accuracy_A_ver

baseline['B (distorted)']['av. PSNR'] = psnr_distort
baseline['B (distorted)']['l2-err'] = l2_err_distort


print("\n# Summary")
print("=========\n")

utility.make_table(
    baseline,
    row_name="baseline",
    out="figures/table_reconstruction_baseline.csv")

print("\nReconstruction methods:")

utility.make_table(
    metrics,
    row_name="method",
    out="figures/table_reconstruction_results.csv")


def plot_metrics(method, **kwargs):
    print(f"\n## {method}")
    utility.plot_metrics(plots[method], **kwargs)
