"""Testing reconstruction methods"""
import os
import sys

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.functional import pad
from torchvision.transforms.functional import crop

import matplotlib.pyplot as plt
# plt.style.use('default')
import numpy as np

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)

USE_DRIVE = True
SCALE_EACH_IM = True

from utils import utility
from utils import methods
from utils import datasets
from utils import debug
from utils import nets
# from utils.haarPsi import haar_psi_numpy
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
parser.add_argument("-dataset", choices=['GMM', 'MNIST', 'CIFAR10', 'MNIST_SVHN', 'SVHN_MNIST'], required=True)
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
parser.add_argument("-run_name", type=str, default="")
parser.add_argument("--silent", action="store_true")
parser.add_argument("-methods", nargs='+', type=str)

if 'ipykernel_launcher' in sys.argv[0]:
    args = parser.parse_args('-dataset MNIST'.split())
    # args = parser.parse_args('-dataset CIFAR10'.split())
    # args = parser.parse_args('-dataset GMM'.split())
    args.inv_steps = 2
    args.size_A = -1
    args.size_B = 8
    # args.size_B = 200
    args.size_C = 8
    # args.size_B = -1
    # args.size_C = -1
    args.batch_size = 8
    args.r_distort_level = -0.04
    args.plot_ideal = True
    args.f_reg = 0
    args.silent = True
    args.seed = 1
    args.seed = 26
    # args.seed = 62

    # args.seed = 55
    # args.nn_resume_train = True
    # args.nn_reset_train = True
    # args.reset_stats = True
    # args.use_std = True
    # args.save_run = True
else:
    args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
utility.seed_everything(args.seed)

if not args.silent:
    print("#", __doc__)
    print("# on", args.dataset)
    print("Hyperparameters:")
    print(utility.dict_to_str(vars(args), '\n'), '\n')
    print(f"Running on '{DEVICE}'\n")


# ======= Create Dataset =======
def get_data_loaders(dataset):
    A, B, C = dataset.get_datasets(size_A=args.size_A, size_B=args.size_B, size_C=args.size_C)

    DATA_A = utility.DataLoaderDevice(A, batch_size=args.batch_size, shuffle=True, device=DEVICE)
    DATA_B = utility.DataLoaderDevice(B, batch_size=args.batch_size, shuffle=True, device=DEVICE)
    DATA_C = utility.DataLoaderDevice(C, batch_size=args.batch_size, shuffle=True, device=DEVICE)
    return DATA_A, DATA_B, DATA_C


if 'SVHN' in args.dataset:
    if args.dataset == 'MNIST_SVHN':
        source_dataset = datasets.MNIST()
        target_dataset = datasets.SVHN()
    elif args.dataset == 'SVHN_MNIST':
        source_dataset = datasets.SVHN()
        target_dataset = datasets.MNIST()
    DATA_A, _, _ = get_data_loaders(source_dataset)
    _, DATA_B, DATA_C = get_data_loaders(target_dataset)

    target_input_shape = target_dataset.input_shape
    source_input_shape = source_dataset.input_shape

    input_shape = target_input_shape
    n_dims = target_dataset.n_dims
    n_classes = source_dataset.n_classes

    dataset = source_dataset
else:
    if args.dataset == 'CIFAR10':
        dataset = datasets.CIFAR10()
    elif args.dataset == 'MNIST':
        dataset = datasets.MNIST()
    elif args.dataset == 'GMM':
        dataset = datasets.MULTIGMM()
    DATA_A, DATA_B, DATA_C = get_data_loaders(dataset)

    input_shape = dataset.input_shape
    n_dims = dataset.n_dims
    n_classes = dataset.n_classes


# dataset.plot()
# plt.tight_layout()
# plt.axis('off')
# plt.savefig(f"figures/{args.dataset}_plot.pdf", bbox_inches='tight')
# plt.show()


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
              resume=nn_resume_training,
              reset=nn_reset_training,
              plot=True,
              use_drive=USE_DRIVE,
              )
net.eval()

if not args.silent:
    utility.print_net_accuracy(net, DATA_A, estimate_epochs=10)

verification_path, verification_net = dataset.verification_net()
if verification_net:
    verification_net.to(DEVICE)
    optimizer = torch.optim.Adam(verification_net.parameters(), lr=nn_lr)
    utility.train(verification_net, DATA_A, criterion, optimizer,
                  model_path=verification_path,
                  epochs=nn_steps,
                  resume=nn_resume_training,
                  reset=nn_reset_training,
                  use_drive=USE_DRIVE,
                  )
    if not args.silent:
        print("verification ", end='')
        utility.print_net_accuracy(
            verification_net, DATA_A, estimate_epochs=10)
print()

# ======= Reconstruction/Distortion Model =======
if args.dataset == 'GMM':
    distort = nets.DistortionModelAffine(args.r_distort_level, n_dims)
    distort.eval()
    distort.to(DEVICE)
    ReconstructionModel = nets.ReconstructionModelAffine
elif 'SVHN' in args.dataset:
    def distort(x): return x
    ReconstructionModel = nets.ReconstructionModelUnet
else:
    distort = nets.DistortionModelConv(args.r_distort_level, input_shape)
    distort.eval()
    distort.to(DEVICE)
    ReconstructionModel = nets.ReconstructionModelResnet


def fig_path_fmt(*name_args, filetype="png"):
    if args.save_run:
        path = "figures"
        if args.run_name:
            path = f"figures/{args.run_name}"
        path = f"{path}/reconstruction_{args.dataset}_{'_'.join(name_args)}.{filetype}".replace(
            ' ', '_')
        save_path, _ = utility.search_drive(path, use_drive=USE_DRIVE)
        return save_path
    return None


show_batch = next(iter(DATA_B))[0][:50].to(DEVICE)
print(show_batch.shape)


if not args.silent and args.dataset != 'GMM':
    utility.im_show(show_batch, fig_path_fmt("ground_truth_full"), scale_each=SCALE_EACH_IM)
    utility.im_show(distort(show_batch), fig_path_fmt("distorted_full"), scale_each=SCALE_EACH_IM)

    print("\nground truth:", flush=True)
    utility.im_show(show_batch[:10], fig_path_fmt("ground_truth"), scale_each=SCALE_EACH_IM)

    print("\ndistorted:")
    utility.im_show(distort(show_batch[:10]), fig_path_fmt("distorted"), scale_each=SCALE_EACH_IM)


Id_mat = torch.eye(n_dims, device=DEVICE).reshape(-1, *input_shape)


@torch.no_grad()
def iqa_metrics(data_loader, transform):
    metrics = {}
    metrics['l2-err'] = ((transform(Id_mat) - Id_mat).norm() / Id_mat.norm()).item()

    if args.dataset == 'CIFAR10' or args.dataset == 'MNIST':
        metrics['PSNR'] = 0
        metrics['SSIM'] = 0
        # metrics['HaarPsi'] = 0

        for inputs, labels in data_loader:
            images = inputs
            restored = transform(images)

            images = utility.to_zero_one(images)
            restored = utility.to_zero_one(restored)

            # for image, restored_image in zip(
            #         images.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
            #         restored.permute(0, 2, 3, 1).squeeze().cpu().numpy()):
            #     metrics['HaarPsi'] += (haar_psi_numpy(image, restored_image)[0]
            #                            / len(data_loader) / len(inputs))

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
                transform(inputs)).item() / len(data_loader)

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


methods = methods.get_methods(DATA_A, net, dataset, args, DEVICE)

for method, loss_fn in methods:
    if args.methods is not None and method not in args.methods:
        # print(f"skipping {method}")
        continue

    if not args.silent:
        print("\n\n\n## Method:", method)

    utility.seed_everything(args.seed)

    if 'SVHN' in args.dataset:
        in_channels = target_dataset.input_shape[0]
        out_channels = source_dataset.input_shape[0]
        reconstruct = ReconstructionModel(in_channels, out_channels)
        reconstruct.forward = debug(reconstruct.forward)
    else:
        reconstruct = ReconstructionModel(args, input_shape, n_dims, n_classes)

    reconstruct.train()
    reconstruct.to(DEVICE)

    def invert_fn(inputs):
        distorted_inputs = distort(inputs)
        if args.dataset == 'SVHN_MNIST':
            distorted_inputs = pad(distorted_inputs, (4, 4, 4, 4))
        reconstructed_inputs = reconstruct(distorted_inputs)
        if args.dataset == 'MNIST_SVHN':
            reconstructed_inputs = crop(reconstructed_inputs, 4, 4, 28, 28)
        return reconstructed_inputs

    def data_loss_fn(data):
        inputs, labels = data

        if args.use_jitter:
            inputs = jitter(inputs)

        data_invert = (invert_fn(inputs), labels)
        info = loss_fn(data_invert)

        if not args.silent:
            iqa = iqa_metrics([data], invert_fn)
            if args.dataset == 'GMM':
                iqa.pop('c-entropy')
                info = {**info, **iqa}
            else:
                info['[IQA metrics] accuracy'] = info['accuracy']
                for k, v in iqa.items():
                    info[f'[IQA metrics] {k}'] = v

            if args.plot_ideal and not 'SVHN' in args.dataset:
                with torch.no_grad():
                    info['reference'] = loss_fn(data)['loss'].item()

        return info

    def callback_fn(epoch, *rgs):
        if args.silent or args.dataset == 'GMM':
            return
        if args.show_after > 0 and epoch % args.show_after == 0:
            print(f"\nepoch {epoch}:", flush=True)
            utility.im_show(invert_fn(show_batch[:10]), fig_path_fmt(
                f"{method}_epoch_{epoch}"), scale_each=SCALE_EACH_IM)

    optimizer = torch.optim.Adam(reconstruct.parameters(), lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    metrics_fig_path = fig_path_fmt(method, filetype="pdf")

    info = utility.invert(DATA_B,
                          data_loss_fn,
                          optimizer,
                          #    scheduler=scheduler,
                          steps=inv_steps,
                          plot=not args.silent,
                          use_amp=args.use_amp,
                          #    grad_norm_fn=grad_norm_fn,
                          callback_fn=callback_fn,
                          track_grad_norm=True,
                          fig_path=metrics_fig_path,
                          )
    plots[method] = info

    # ======= Result =======
    reconstruct.eval()

    if not args.silent and args.dataset != 'GMM':
        print("Inverted:")
        # if len(show_batch) != len(B):
        #     print(f"{len(show_batch)} / {len(B)} ")
        utility.im_show(invert_fn(show_batch), fig_path_fmt(
            f"{method}_epoch_{inv_steps}_full"), scale_each=SCALE_EACH_IM)

    # Loss
    if not args.silent:
        loss = info['loss'].values[-1]
        print("Results:")
        print(f"\tloss: {loss:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, DATA_B, inputs_pre_fn=invert_fn)
    accuracy_val = utility.net_accuracy(
        net, DATA_C, inputs_pre_fn=invert_fn)
    metrics[method]['acc'] = accuracy
    metrics[method]['acc(val)'] = accuracy_val
    if not args.silent:
        print(f"\tnn accuracy: {accuracy * 100:.1f} %")
        print(f"\tnn validation set accuracy: {accuracy_val * 100:.1f} %")

    if verification_net:
        accuracy_ver = utility.net_accuracy(
            verification_net, DATA_C, inputs_pre_fn=invert_fn)
        metrics[method]['acc(ver)'] = accuracy_ver
        if not args.silent:
            print(f"\tnn verification accuracy: {accuracy_ver * 100:.1f} %")

    iqa_invert = iqa_metrics(DATA_B, invert_fn)

    for k, v in iqa_invert.items():
        metrics[method][k] = v
        if not args.silent:
            print(f"\taverage {k}: {v:.3f} | (before: {iqa_distort[k]:.3f})")


baseline = defaultdict(dict)


accuracy_A = utility.net_accuracy(net, DATA_A)
accuracy_B = utility.net_accuracy(net, DATA_B) if 'SVHN' not in args.dataset else '-'
accuracy_C = utility.net_accuracy(net, DATA_C) if 'SVHN' not in args.dataset else '-'

accuracy_B_pert = utility.net_accuracy(net, DATA_B, inputs_pre_fn=distort)
accuracy_C_pert = utility.net_accuracy(net, DATA_C, inputs_pre_fn=distort)


if verification_net:
    accuracy_A_ver = utility.net_accuracy(verification_net, DATA_A) if 'SVHN' not in args.dataset else '-'
    accuracy_B_ver = utility.net_accuracy(verification_net, DATA_B) if 'SVHN' not in args.dataset else '-'
    accuracy_C_pert_ver = utility.net_accuracy(
        verification_net, DATA_C, inputs_pre_fn=distort) if 'SVHN' not in args.dataset else '-'


baseline['Source A (original)']['acc'] = accuracy_B
baseline['Source A (original)']['acc(val)'] = accuracy_C

baseline['Source A (perturbed)']['acc'] = accuracy_B_pert
baseline['Source A (perturbed)']['acc(val)'] = accuracy_C_pert

if verification_net:
    baseline['Source A (perturbed)']['acc(ver)'] = accuracy_C_pert_ver
    baseline['Source A (original)']['acc(ver)'] = accuracy_B_ver
    baseline['Target B']['acc(ver)'] = accuracy_A_ver

for k, v in reversed(sorted(iqa_distort.items())):
    baseline['Source A (perturbed)'][k] = v

if args.dataset == 'GMM':
    baseline['Target B']['c-entropy'] = iqa_metrics(DATA_A, lambda x: x)['c-entropy']
    baseline['Source A (original)']['c-entropy'] = iqa_metrics(DATA_B, lambda x: x)['c-entropy']

baseline['Target B']['acc'] = accuracy_A


if not args.silent:
    print("\n# Summary")
    print("=========\n")

    table_path = fig_path_fmt("baseline", filetype="csv")
    utility.make_table(baseline, row_name="baseline", out=table_path)

    print("\nReconstruction methods:")

    table_path = fig_path_fmt("results", filetype="csv")
    utility.make_table(metrics, row_name="method", out=table_path)
