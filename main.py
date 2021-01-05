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
parser = argparse.ArgumentParser(description="Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'GMM', 'MNIST'], required=True)
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_std", action="store_true")
parser.add_argument("--use_jitter", action="store_true")
parser.add_argument("--plot_ideal", action="store_true")
parser.add_argument("--normalize_images", action="store_true")
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
parser.add_argument("-show_after", type=int, default=20)
parser.add_argument("-perturb_strength", type=float, default=0.3)

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
    args.perturb_strength = 0.3
    # args.batch_size = 64
    # # args.size_B = 10
    # # args.n_random_projections = 1024
    args.inv_lr = 0.01
    args.f_stats = 0.001
    # args.perturb_strength = 0.5

    # args = parser.parse_args('-dataset CIFAR10'.split())
    # args.inv_steps = 1
    # args.batch_size = 64
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

# ======= Perturbation =======


class perturb_model(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 3
        nch = input_shape[0]
        lambd = args.perturb_strength

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

    def forward(self, inputs):
        outputs = inputs
        with torch.no_grad():
            outputs = outputs + self.noise
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
        return outputs


perturb = perturb_model()
perturb.eval()
perturb.to(DEVICE)

# ======= Preprocessing Model =======


class preprocessing_model(nn.Module):
    def __init__(self):
        super().__init__()

        nch = input_shape[0]
        n_chan_inner = 8

        self.block_layer = nn.Sequential(
            nets.ResidualBlock(nch, n_chan_inner, 1, bias=False),
            nets.ResidualBlock(n_chan_inner, nch, 1, bias=False),
        )
        # kernel_size = 3
        # self.conv1 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv2 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv3 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv4 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.shift = nn.Parameter(torch.zeros(input_shape).unsqueeze(0))

    def forward(self, inputs):
        return self.block_layer(inputs)
        # outputs = inputs
        # outputs = outputs + self.shift
        # outputs = self.conv1(outputs)
        # outputs = self.conv2(outputs)
        # outputs = self.conv3(outputs)
        # outputs = self.conv4(outputs)
        # return outputs


# ======= Setup Methods =======
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

if not 'ipykernel_launcher' in sys.argv[0]:
    utility.print_net_accuracy(net, DATA_A)

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
        utility.print_net_accuracy(verifier_net, DATA_A)


# ======= NN Project =======
# NOTE: when using bn_layers, use inputs from hook
# net_layers = utility.get_bn_layers(net)
net_layers = utility.get_child_modules(net)[:-1]
layer_activations = [None] * len(net_layers)
net_last_outputs = None


def layer_hook_wrapper(idx):
    def hook(_module, _inputs, outputs):
        layer_activations[idx] = outputs
    return hook


for l, layer in enumerate(net_layers):
    layer.register_forward_hook(layer_hook_wrapper(l))


def project_NN(data):
    global net_last_outputs
    inputs, labels = data
    net_last_outputs = net(inputs)
    return layer_activations[-1]
    # return net_last_outputs


def project_NN_all(data):
    global net_last_outputs
    inputs, labels = data
    net_last_outputs = net(inputs)
    return [inputs] + layer_activations


# ======= Random Projections =======
rp_hash = f"{n_random_projections}"
path_RP = os.path.join(MODELDIR, f"RP-{rp_hash}")


@utility.store_data
def random_projections():
    RP = torch.randn((n_dims, n_random_projections)).to(DEVICE)
    RP = RP / RP.norm(2, dim=0)
    return RP


# for reproducibility
RP = random_projections(  # pylint: disable=unexpected-keyword-arg
    path=path_RP, map_location=DEVICE, use_drive=USE_DRIVE)


def get_input(data):
    return data[0]


mean_A, std_A = utility.collect_stats(
    DATA_A, get_input, n_classes, class_conditional=False, std=True, keepdim=True,
    path=stats_path.format('inputs'), device=DEVICE, use_drive=USE_DRIVE)
mean_A_C, std_A_C = utility.collect_stats(
    DATA_A, get_input, n_classes, class_conditional=True, std=True, keepdim=True,
    path=stats_path.format('inputs-CC'), device=DEVICE, use_drive=USE_DRIVE)

# min_A, max_A = utility.collect_min_max(
#     DATA_A, path=stats_path.format('min-max'), device=DEVICE, use_drive=USE_DRIVE)


def project_RP(data):
    X, Y = data
    return (X - mean_A).reshape(-1, n_dims) @ RP


def project_RP_CC(data):
    X, Y = data
    X_proj_C = None
    for c in range(n_classes):
        mask = Y == c
        X_proj_c = (X[mask] - mean_A_C[c]).reshape(-1, n_dims) @ RP
        if X_proj_C is None:
            X_proj_C = torch.empty((X.shape[0], n_random_projections),
                                   dtype=X_proj_c.dtype, device=X.device)
        X_proj_C[mask] = X_proj_c
    return X_proj_C


mean_RP_A, std_RP_A = utility.collect_stats(
    DATA_A, project_RP, n_classes, class_conditional=False, std=True, keepdim=True,
    path=stats_path.format(f"RP-{rp_hash}"), device=DEVICE, use_drive=USE_DRIVE)
mean_RP_A_C, std_RP_A_C = utility.collect_stats(
    DATA_A, project_RP_CC, n_classes, class_conditional=True, std=True, keepdim=True,
    path=stats_path.format(f"RP-CC-{rp_hash}"), device=DEVICE, use_drive=USE_DRIVE)

# Random ReLU Projections
f_rp_relu = 1 / 2
relu_bias = mean_RP_A + f_rp_relu * std_RP_A * torch.randn_like(mean_RP_A)
relu_bias_C = (mean_RP_A_C +
               f_rp_relu * std_RP_A_C * torch.randn_like(mean_RP_A_C)).squeeze()


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
    return (torch.norm(diff1) + torch.norm(diff2) +
            torch.norm(diff3) + torch.norm(diff4))


# @debug
def loss_stats(stats_a, stats_b):
    if not isinstance(stats_a, list):
        stats_a, stats_b = [stats_a], [stats_b]
    assert len(stats_a) == len(stats_b), "lists need to be of same length"
    num_stats = len(stats_a)
    loss = torch.tensor(0).float().to(DEVICE)
    info = {}
    for i, ((ma, sa), (mb, sb)) in enumerate(zip(stats_a, stats_b)):
        if ma.ndim == 1:
            loss_m = (ma.squeeze() - mb.squeeze()).norm()
            loss_s = (sa.squeeze() - sb.squeeze()).norm()
        else:   # class conditional
            if np.prod(ma.shape) == ma.shape[0] or np.prod(mb.shape) == mb.shape[0]:
                loss_m = (ma.squeeze() - mb.squeeze()).abs().mean()
                loss_s = (sa.squeeze() - sb.squeeze()).abs().mean()
            else:  # multiple features
                loss_m = (ma.squeeze() - mb.squeeze()).norm(dim=1).mean()
                loss_s = (sa.squeeze() - sb.squeeze()).norm(dim=1).mean()
        if num_stats > 1:
            info[f'[stats losses means] {i}'] = loss_m.item()
            info[f'[stats losses vars] {i}'] = loss_s.item()
        else:
            info[f'[stats losses] mean'] = loss_m.item()
            info[f'[stats losses] var'] = loss_s.item()
        loss += loss_m + loss_s
    return loss, info


f_crit = args.f_crit
f_reg = args.f_reg
f_stats = args.f_stats


def loss_fn_wrapper(name, project, class_conditional):
    _name = name.replace(' ', '-')
    if "RP" in _name:
        _name = f"{_name}-{rp_hash}"

    stats_A = utility.collect_stats(
        DATA_A, project, n_classes, class_conditional,
        std=STD, path=stats_path.format(_name), device=DEVICE, use_drive=USE_DRIVE)

    def _loss_fn(data, project=project, class_conditional=class_conditional):
        global net_last_outputs
        net_last_outputs = None

        inputs, labels = data

        info = {}
        loss = torch.tensor(0).float().to(DEVICE)

        if f_reg:
            loss_reg = f_reg * regularization(inputs)
            info['[losses] reg'] = loss_reg.item()
            loss += loss_reg

        if f_stats:
            outputs = project(data)
            stats = utility.get_stats(
                outputs, labels, n_classes, class_conditional=class_conditional, std=STD)
            cost_stats, info_stats = loss_stats(stats_A, stats)
            cost_stats *= f_stats
            info = {**info, **info_stats}
            info['[losses] stats'] = cost_stats.item()
            loss += cost_stats

        if f_crit:
            if net_last_outputs is None:
                net_last_outputs = net(inputs)
            loss_crit = f_crit * criterion(net_last_outputs, labels)
            info['[losses] crit'] = loss_crit.item()
            loss += loss_crit

            info[':mean: accuracy'] = utility.count_correct(
                net_last_outputs, labels) / len(labels)

        info['loss'] = loss
        return info
        # return loss
    return name, _loss_fn


methods = [
    loss_fn_wrapper(
        name="NN",
        project=project_NN,
        class_conditional=False,
    ),
    loss_fn_wrapper(
        name="NN CC",
        project=project_NN,
        class_conditional=True,
    ),
    loss_fn_wrapper(
        name="NN ALL",
        project=project_NN_all,
        class_conditional=False,
    ),
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
        name="NN ALL + RP CC",
        project=combine(project_NN_all, project_RP_CC),
        class_conditional=True,
    ),
]


def im_show(batch):
    with torch.no_grad():
        img_grid = torchvision.utils.make_grid(
            batch.cpu(), nrow=10, normalize=args.normalize_images, scale_each=True)
        plt.figure(figsize=(16, (1.6 * len(batch)) // 10))
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()


show_batch = next(iter(DATA_B))[0][:10].to(DEVICE)

print("\nground truth:", flush=True)
im_show(show_batch)

print("\nperturbed:")
im_show(perturb(show_batch))

# ======= Optimize =======
metrics = defaultdict(dict)


def jitter(inputs):
    shifts = tuple(torch.randint(low=-2, high=2, size=(2,)))
    return torch.roll(inputs, shifts=shifts, dims=(2, 3))


def grad_norm_fn(x):
    return min(x, 10)  # torch.sqrt(x) if x > 1 else x


for method, loss_fn in methods:
    print("\n## Method:", method)

    preprocess = preprocessing_model()
    preprocess.train()
    preprocess.to(DEVICE)

    def invert_fn(inputs):
        return preprocess(perturb(inputs))

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
        if epoch % args.show_after == 0 and epoch > 0:
            print(f"\nepoch {epoch}:", flush=True)
            im_show(invert_fn(show_batch))
            print(flush=True)

    optimizer = torch.optim.Adam(preprocess.parameters(), lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = inversion.invert(DATA_B,
                            data_loss_fn,
                            optimizer,
                            #    scheduler=scheduler,
                            steps=inv_steps,
                            plot=True,
                            use_amp=args.use_amp,
                            #    grad_norm_fn=grad_norm_fn,
                            callback_fn=callback_fn,
                            track_grad_norm=True,
                            )

    # ======= Result =======
    preprocess.eval()

    print("Inverted:")
    if args.normalize_images:
        print("(normalized)")
    im_show(invert_fn(next(iter(DATA_B))[0]))

    print("Results:")

    # Loss
    # loss = accumulate_fn(DATA_B, loss_fn)
    loss = info['loss'][-1]
    print(f"\tloss: {loss:.3f}")

    # PSNR
    psnr = utility.average_psnr(DATA_B, invert_fn)
    psnr_perturb = utility.average_psnr(DATA_B, perturb)
    print(f"\taverage PSNR: {psnr:.3f} | {psnr_perturb:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE).reshape(-1, *input_shape)
    l2_err = (invert_fn(Id) - Id).norm().item() / Id.norm().item()
    l2_err_perturb = (perturb(Id) - Id).norm().item() / Id.norm().item()
    print(
        f"\trel. l2 reconstruction error: {l2_err:.3f} | {l2_err_perturb:.3f}")

    # NN Accuracy
    accuracy = utility.net_accuracy(net, DATA_B, inputs_pre_fn=invert_fn)
    accuracy_val = utility.net_accuracy(
        net, DATA_B_val, inputs_pre_fn=invert_fn)
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
    metrics[method]['loss'] = loss

baseline = defaultdict(dict)


accuracy_A = utility.net_accuracy(net, DATA_A)
accuracy_B = utility.net_accuracy(net, DATA_B)
accuracy_B_val = utility.net_accuracy(
    net, DATA_B_val)

accuracy_B_pert = utility.net_accuracy(
    net, DATA_B, inputs_pre_fn=perturb)
accuracy_B_val_pert = utility.net_accuracy(
    net, DATA_B_val, inputs_pre_fn=perturb)

if verifier_net:
    accuracy_A_ver = utility.net_accuracy(
        verifier_net, DATA_A)
    accuracy_B_ver = utility.net_accuracy(
        verifier_net, DATA_B)
    accuracy_B_pert_ver = utility.net_accuracy(
        verifier_net, DATA_B, inputs_pre_fn=perturb)

baseline['B (original)']['acc'] = accuracy_B
baseline['B (original)']['acc(val)'] = accuracy_B_val

baseline['B (perturbed)']['acc'] = accuracy_B_pert
baseline['B (perturbed)']['acc(val)'] = accuracy_B_val_pert

baseline['A']['acc'] = accuracy_A

if verifier_net:
    baseline['B (perturbed)']['acc(ver)'] = accuracy_B_pert_ver
    baseline['B (original)']['acc(ver)'] = accuracy_B_ver
    baseline['A']['acc(ver)'] = accuracy_A_ver

baseline['B (perturbed)']['av. PSNR'] = psnr_perturb
baseline['B (perturbed)']['l2-err'] = l2_err_perturb


print("\n# Summary")
print("=========\n")

utility.print_tabular(baseline, row_name="baseline")

print("\nReconstruction methods:")

utility.print_tabular(metrics, row_name="method")
