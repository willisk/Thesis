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
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt

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
parser.add_argument("--use_var", action="store_true")
parser.add_argument("-nn_lr", type=float, default=0.01)
parser.add_argument("-nn_steps", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_random_projections", type=int, default=256)
parser.add_argument("-inv_lr", type=float, default=0.1)
parser.add_argument("-inv_steps", type=int, default=100)
parser.add_argument("-f_reg", type=float, default=0.001)
parser.add_argument("-f_crit", type=float, default=1)
parser.add_argument("-f_stats", type=float, default=1)
parser.add_argument("-size_A", type=int, default=-1)
parser.add_argument("-size_B", type=int, default=64)
parser.add_argument("-perturb_strength", type=float, default=0.03)
parser.add_argument("-preprocessing_depth", type=int, default=2)

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
    args.nn_steps = 5
    args.inv_steps = 100
    args.batch_size = 64
    # args.size_B = 10
    # args.n_random_projections = 1024
    args.inv_lr = 0.05
    args.perturb_strength = 0.5

    # args = parser.parse_args('-dataset CIFAR10'.split())
    # args.inv_steps = 1
    # args.batch_size = 64
    args.seed = 10

    args.size_B = 64
    # args.nn_resume_train = True
    # args.nn_reset_train = True
    # args.use_var = True
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


def data_loader(D):
    batch_size = args.batch_size if args.batch_size != -1 else len(D)
    if args.seed != -1:
        return DataLoader(D, batch_size=batch_size, shuffle=True,
                          worker_init_fn=lambda x: np.random.seed(args.seed))
    return DataLoader(D, batch_size=batch_size, shuffle=True)


DATA_A = data_loader(A)
DATA_B = data_loader(B)
DATA_B_val = data_loader(B_val)

input_shape = dataset.input_shape
n_dims = dataset.n_dims
n_classes = dataset.n_classes


STD = not args.use_var
stats_path = os.path.join(MODELDIR, "stats_{}.pt")
# ======= Perturbation =======


class perturb_model(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 3
        nch = input_shape[0]
        self.conv1 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        self.noise = nn.Parameter(
            torch.randn(input_shape).unsqueeze(
                0) * args.perturb_strength)

    def forward(self, inputs):
        outputs = inputs
        with torch.no_grad():
            outputs = outputs + self.noise
            outputs = self.conv1(outputs)
            outputs = self.conv2(outputs)
        return outputs


perturb = perturb_model()
perturb.to(DEVICE)


# ======= Preprocessing Model =======
class preprocessing_model(nn.Module):
    def __init__(self):
        super().__init__()
        nch = input_shape[0]
        # block_depth = args.preprocessing_depth
        # self.block_layer = nn.Sequential(*[
        #     nets.ResidualBlock(nch, nch, 1) for _ in range(block_depth)])
        n_chan_inner = 8
        self.block_layer = nn.Sequential(
            nets.ResidualBlock(nch, n_chan_inner, 1, bias=True),
            nets.ResidualBlock(n_chan_inner, nch, 1, bias=True),
        )
        # kernel_size = 3
        # self.conv1 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv2 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv3 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.conv4 = nn.Conv2d(nch, nch, kernel_size, padding=1)
        # self.shift = nn.Parameter(torch.zeros(input_shape).unsqueeze(0))

    def forward(self, inputs):
        # outputs = self.block_layer(inputs)
        # debug(outputs)
        # return outputs
        return self.block_layer(inputs)
        # outputs = inputs
        # outputs = outputs + self.shift
        # outputs = self.conv1(outputs)
        # outputs = self.conv2(outputs)
        # outputs = self.conv3(outputs)
        # outputs = self.conv4(outputs)
        # return outputs


# def perturb(X):
#     X_shape = X.shape
#     X = X.reshape(-1, n_dims)
#     out = X @ perturb_matrix + perturb_shift
#     return out.reshape(X_shape)
# M = torch.eye(n_dims, requires_grad=True, device=DEVICE)
# b = torch.zeros((n_dims), requires_grad=True, device=DEVICE)
# def preprocessing_fn(X):
#     X_shape = X.shape
#     X = X.reshape(-1, n_dims)
#     return (X @ M + b).reshape(X_shape)
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


def project_NN(data):
    global net_last_outputs
    inputs, labels = data
    net_last_outputs = net(inputs)
    return net_last_outputs


def project_NN_all(data):
    global net_last_outputs
    inputs, labels = data
    net_last_outputs = net(inputs)
    return [inputs] + layer_activations


# ======= Random Projections =======
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)


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


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXX Random projections need not always be the same? verify seed
mean_RP_A, std_RP_A = utility.collect_stats(
    DATA_A, project_RP, n_classes, class_conditional=False, std=True, keepdim=True,
    path=stats_path.format('RP'), device=DEVICE, use_drive=USE_DRIVE)
mean_RP_A_C, std_RP_A_C = utility.collect_stats(
    DATA_A, project_RP_CC, n_classes, class_conditional=True, std=True, keepdim=True,
    path=stats_path.format('RP-CC'), device=DEVICE, use_drive=USE_DRIVE)

# Random ReLU Projections
f_rp_relu = 1 / 2
relu_bias = mean_RP_A + f_rp_relu * std_RP_A * torch.randn_like(mean_RP_A)
relu_bias_C = (mean_RP_A_C +
               f_rp_relu * std_RP_A_C * torch.randn_like(mean_RP_A_C))


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
    loss = torch.tensor(0).float().to(DEVICE)
    for (ma, sa), (mb, sb) in zip(stats_a, stats_b):
        if ma.ndim == 1:
            loss += (ma.squeeze() - mb.squeeze()).norm()
            loss += (sa.squeeze() - sb.squeeze()).norm()
        else:
            if np.prod(ma.shape) == ma.shape[0]:    # one feature
                loss += (ma.squeeze() - mb.squeeze()).abs().mean()
                loss += (sa.squeeze() - sb.squeeze()).abs().mean()
            else:
                loss += (ma.squeeze() - mb.squeeze()).norm(dim=1).mean()
                loss += (sa.squeeze() - sb.squeeze()).norm(dim=1).mean()
    return loss
    # return sum(
    #     (ma.squeeze() - mb.squeeze()).norm() +
    #     (sa.squeeze() - sb.squeeze()).norm() if ma.ndim == 1 else
    #     (ma.squeeze() - mb.squeeze()).norm(dim=1).mean() +  # class_conditional
    #     (sa.squeeze() - sb.squeeze()).norm(dim=1).mean()
    #     for (ma, sa), (mb, sb) in zip(stats_a, stats_b))  # / len(stats_a)


debug.silent = False

f_crit = args.f_crit
f_reg = args.f_reg
f_stats = args.f_stats


def loss_fn_wrapper(name, project, class_conditional):
    _name = name.replace(' ', '-')
    stats_A = utility.collect_stats(
        DATA_A, project, n_classes, class_conditional,
        std=STD, path=stats_path.format(_name), device=DEVICE, use_drive=USE_DRIVE)

    def _loss_fn(data, project=project, class_conditional=class_conditional):
        global net_last_outputs
        net_last_outputs = None

        inputs, labels = data
        outputs = project(data)

        stats = utility.get_stats(
            outputs, labels, n_classes, class_conditional=class_conditional, std=STD)

        loss = torch.tensor(0).float().to(DEVICE)
        loss += f_stats * loss_stats(stats, stats_A) if f_stats else 0
        loss += f_reg * regularization(inputs) if f_reg else 0

        if f_crit:
            if net_last_outputs is None:
                net_last_outputs = net(inputs)
            loss += f_crit * criterion(net_last_outputs, labels)
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
    # loss_fn_wrapper(
    #     name="NN ALL",
    #     project=project_NN_all,
    #     class_conditional=False,
    # ),
    # loss_fn_wrapper(
    #     name="NN ALL CC",
    #     project=project_NN_all,
    #     class_conditional=True,
    # ),
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
    # loss_fn_wrapper(
    #     name="combined",
    #     project=combine(project_NN_all, project_RP_CC),
    #     class_conditional=True,
    # ),


]


def im_show(batch):
    with torch.no_grad():
        img_grid = torchvision.utils.make_grid(
            batch.cpu(), nrow=5, normalize=True, scale_each=True)
        plt.figure(figsize=(16, 4))
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.show()


# DATA_B = [next(iter(DATA_B))]
show_batch = next(iter(DATA_B))[0][:10].to(DEVICE)

print("\nground truth:", flush=True)
im_show(show_batch)

print("\nperturbed:")
im_show(perturb(show_batch))

# ======= Optimize =======
metrics = defaultdict(dict)


def grad_norm_fn(x):
    return min(x, 10)  # torch.sqrt(x) if x > 1 else x


for method, loss_fn in methods:
    print("\n## Method:", method)

    preprocess = preprocessing_model()
    preprocess.train()
    preprocess.to(DEVICE)

    def data_loss_fn(data):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = perturb(inputs)
        outputs = preprocess(outputs)
        data = (outputs, labels)
        return loss_fn(data)

    optimizer = torch.optim.Adam(preprocess.parameters(), lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = inversion.invert(DATA_B,
                            data_loss_fn,
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
                            )

    # ======= Result =======
    preprocess.eval()

    def invert_fn(inputs):
        return preprocess(perturb(inputs))

    print("Inverted:")
    im_show(invert_fn(show_batch))

    print("Results:")

    # Loss
    # loss = accumulate_fn(DATA_B, loss_fn)
    loss = info['loss'][-1]
    print(f"\tloss: {loss:.3f}")

    # L2 Reconstruction Error
    Id = torch.eye(n_dims, device=DEVICE).reshape(-1, *input_shape)
    l2_err_perturb = (perturb(Id) - Id).norm().item() / Id.norm().item()
    l2_err = (invert_fn(Id) - Id).norm().item() / Id.norm().item()
    print(
        f"\trel. l2 reconstruction error: {l2_err:.3f} / {l2_err_perturb:.3f}")

    # PSNR
    psnr = average_psnr(show_batch, invert_fn(show_batch))
    print(
        f"\tPSNR: {psnr_B:.3f}")

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
    metrics[method]['l2-err'] = l2_err
    metrics[method]['loss'] = loss
    metrics[method]['psnr'] = psnr

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


print("\n# Summary")
print("=========\n")

utility.print_tabular(baseline, row_name="baseline")

print("\nReconstruction methods:")

utility.print_tabular(metrics, row_name="method")
