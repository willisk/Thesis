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

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)

USE_DRIVE = True

from utils import utility
from utils import methods
from utils import datasets
from utils import debug
from utils import nets
from utils.haarPsi import haar_psi_numpy
from pytorch_lightning.metrics.functional import ssim

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

from utils.debug import debug


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'MNIST', 'GMM'], required=True)
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
parser.add_argument("-size_C", type=int, default=1024)
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
parser.add_argument("--reset_stats", action="store_true")
parser.add_argument("--save_run", action="store_true")

# # GMM
# parser.add_argument("-g_modes", type=int, default=3)
# parser.add_argument("-g_scale_mean", type=float, default=2)
# parser.add_argument("-g_scale_cov", type=float, default=20)
# parser.add_argument("-g_mean_shift", type=float, default=0)

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset MNIST'.split())
    # args = parser.parse_args('-dataset CIFAR10'.split())
    args = parser.parse_args('-dataset GMM'.split())
    args.inv_steps = 2
    args.size_A = -1
    args.size_B = 8
    args.size_C = 8
    args.batch_size = 8
    args.r_distort_level = 0.1
    args.plot_ideal = True
    args.f_reg = 0
    # args.nn_resume_train = True
    # args.nn_reset_train = True
    # args.reset_stats = True
    # args.use_std = True
    # args.save_run = True
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
elif args.dataset == 'GMM':
    dataset = datasets.MULTIGMM()

A, B, C = dataset.get_datasets(
    size_A=args.size_A, size_B=args.size_B, size_C=args.size_C)


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


if args.dataset == 'GMM':
    class DistortionModel(nn.Module):
        def __init__(self, lambd):
            super().__init__()
            self.noise = nn.Parameter(
                torch.randn((n_dims)).unsqueeze(0)) * lambd
            self.linear = (torch.eye(n_dims)
                           + torch.randn((n_dims, n_dims)) * lambd)

        @torch.no_grad()
        def forward(self, inputs):
            outputs = inputs + self.noise
            return outputs @ self.linear
else:
    class DistortionModel(nn.Module):
        def __init__(self, lambd):
            super().__init__()

            kernel_size = 3
            nch = input_shape[0]

            self.conv1 = nn.Conv2d(nch, nch, kernel_size,
                                   padding=1, padding_mode='reflect')
            self.conv2 = nn.Conv2d(nch, nch, kernel_size,
                                   padding=1, padding_mode='reflect')
            self.noise = nn.Parameter(torch.randn(input_shape).unsqueeze(0))

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

distort = DistortionModel(args.r_distort_level)
distort.eval()
distort.to(DEVICE)

# ======= Reconstruction Model =======
if args.dataset == 'GMM':
    class ReconstructionModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.bias = nn.Parameter(
                torch.zeros((n_dims)).unsqueeze(0))
            self.linear = nn.Parameter(
                torch.eye(n_dims) + torch.randn((n_dims, n_dims)) / np.sqrt(n_dims))

        def forward(self, inputs):
            return inputs @ self.linear + self.bias
else:
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


methods = methods.get_methods(DATA_A, net, dataset, args, DEVICE)


def fig_path_fmt(name, filetype="png"):
    if args.save_run:
        path = f"figures/reconstruction_{args.dataset}_{name}.{filetype}".replace(
            ' ', '_')
        save_path, _ = utility.search_drive(path, use_drive=USE_DRIVE)
        return save_path
    return None


show_batch = next(iter(DATA_B))[0][:50].to(DEVICE)


if args.dataset != 'GMM':
    utility.im_show(show_batch,
                    fig_path_fmt(f"ground_truth_full"))
    utility.im_show(distort(show_batch),
                    fig_path_fmt(f"distorted_full"))

    print("\nground truth:", flush=True)
    utility.im_show(show_batch[:10],
                    fig_path_fmt(f"ground_truth"))

    print("\ndistorted:")
    utility.im_show(distort(show_batch[:10]),
                    fig_path_fmt(f"distorted"))


Id_mat = torch.eye(n_dims, device=DEVICE).reshape(-1, *input_shape)


@torch.no_grad()
def iqa_metrics(data_loader, transform):
    metrics = {}
    metrics['l2-err'] = ((transform(Id_mat) -
                          Id_mat).norm() / Id_mat.norm()).item()

    if args.dataset == 'CIFAR10' or args.dataset == 'MNIST':
        metrics['PSNR'] = 0
        metrics['SSIM'] = 0
        metrics['HaarPsi'] = 0

        for inputs, labels in data_loader:
            images = inputs
            restored = transform(images)

            images = utility.to_zero_one(images)
            restored = utility.to_zero_one(restored)

            for image, restored_image in zip(
                    images.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                    restored.permute(0, 2, 3, 1).squeeze().cpu().numpy()):
                metrics['HaarPsi'] += (haar_psi_numpy(image, restored_image)[0]
                                       / len(data_loader) / len(inputs))

            if images.shape[1] == 3:
                images = utility.rbg_to_luminance(images)
                restored = utility.rbg_to_luminance(restored)

            metrics['PSNR'] += utility.psnr(
                images, restored).mean().item() / len(data_loader)
            metrics['SSIM'] += (1 + ssim(
                images, restored).item()) / 2 / len(data_loader)  # default: mean, scale to [0, 1]

    if args.dataset == 'GMM':
        metrics['c-entropy'] = 0
        for inputs, labels in data_loader:
            metrics['c-entropy'] += dataset.cross_entropy(
                transform(inputs)) / len(data_loader)

    return metrics


iqa_distort = iqa_metrics(DATA_B, distort)

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


for method, loss_fn in methods:
    # if method == "NN ALL":
    #     break
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

        data_invert = (invert_fn(inputs), labels)
        info = loss_fn(data_invert)

        iqa = iqa_metrics([data], invert_fn)
        if args.dataset == 'GMM':
            iqa.pop('c-entropy')
            info = {**info, **iqa}
        else:
            info['[IQA metrics] accuracy'] = info['accuracy']
            for k, v in iqa.items():
                info[f'[IQA metrics] {k}'] = v

        if args.plot_ideal:
            with torch.no_grad():
                info['reference'] = loss_fn(data)['loss'].item()

        return info

    def callback_fn(epoch, metrics):
        if args.dataset == 'GMM':
            return
        if epoch % args.show_after == 0:
            print(f"\nepoch {epoch}:", flush=True)
            utility.im_show(invert_fn(show_batch[:10]),
                            fig_path_fmt(f"{method}_epoch_{epoch}"))

    optimizer = torch.optim.Adam(reconstruct.parameters(), lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    metrics_fig_path = fig_path_fmt(f"{method}", "pdf")

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
                          fig_path=metrics_fig_path,
                          )
    plots[method] = info

    # ======= Result =======
    reconstruct.eval()

    if args.dataset != 'GMM':
        print("Inverted:")
        if len(show_batch) != len(B):
            print(f"{len(show_batch)} / {len(B)} ")
        utility.im_show(invert_fn(show_batch), fig_path_fmt(
            f"{method}_epoch_{inv_steps}_full"))

    print("Results:")

    # Loss
    loss = info['loss'].values[-1]
    print(f"\tloss: {loss:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, DATA_B, inputs_pre_fn=invert_fn)
    accuracy_val = utility.net_accuracy(
        net, DATA_C, inputs_pre_fn=invert_fn)
    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")
    print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    if verifier_net:
        accuracy_ver = utility.net_accuracy(
            verifier_net, DATA_B, inputs_pre_fn=invert_fn)
        metrics[method]['acc(ver)'] = accuracy_ver
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")

    iqa_invert = iqa_metrics(DATA_B, invert_fn)

    for k, v in iqa_invert.items():
        metrics[method][k] = v
        print(f"\taverage {k}: {v:.3f} | (before: {iqa_distort[k]:.3f})")


baseline = defaultdict(dict)


accuracy_A = utility.net_accuracy(net, DATA_A)
accuracy_B = utility.net_accuracy(net, DATA_B)
accuracy_C = utility.net_accuracy(net, DATA_C)

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


baseline['Target A']['acc'] = accuracy_A

baseline['Source B (original)']['acc'] = accuracy_B
baseline['Source B (original)']['acc(val)'] = accuracy_C

baseline['Source B (distorted)']['acc'] = accuracy_B_pert
baseline['Source B (distorted)']['acc(val)'] = accuracy_C_pert

if verifier_net:
    baseline['Source B (distorted)']['acc(ver)'] = accuracy_B_pert_ver
    baseline['Source B (original)']['acc(ver)'] = accuracy_B_ver
    baseline['Target A']['acc(ver)'] = accuracy_A_ver

for k, v in reversed(sorted(iqa_distort.items())):
    baseline['Source B (distorted)'][k] = v

if args.dataset == 'GMM':
    baseline['Target A']['c-entropy'] = iqa_metrics(
        DATA_A, lambda x: x)['c-entropy']
    baseline['Source B (original)']['c-entropy'] = iqa_metrics(
        DATA_B, lambda x: x)['c-entropy']


print("\n# Summary")
print("=========\n")


table_path = fig_path_fmt("baseline", "csv")
utility.make_table(baseline, row_name="baseline", out=table_path)

print("\nReconstruction methods:")

table_path = fig_path_fmt("results", "csv")
utility.make_table(metrics, row_name="method", out=table_path)


def plot_metrics(method, **kwargs):
    print(f"\n## {method}")
    utility.plot_metrics(plots[method], **kwargs)
