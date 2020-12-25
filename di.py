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
import debug

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

from debug import debug


# ======= Arg Parse =======
parser = argparse.ArgumentParser(description="GMM Reconstruction Tests")
parser.add_argument(
    "-dataset", choices=['CIFAR10', 'MNIST'], required=True)
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
# parser.add_argument("-f_input", type=float, default=1)

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
    args.use_var = True

    args.n_random_projections = 1024
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

if args.dataset == 'CIFAR10':
    dataset = datasets.CIFAR10(load_data=True)
elif args.dataset == 'MNIST':
    dataset = datasets.MNIST()

MODELDIR = dataset.data_dir


A, B, B_val = dataset.get_datasets()


def data_loader(D):
    return DataLoader(D, batch_size=64, shuffle=True)


DATA_A = data_loader(A)

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
              use_drive=USE_DRIVE,
              )
net.eval()


# ======= NN Project =======
# NOTE: when using bn_layers, use inputs from hook
# net_layers = utility.get_bn_layers(net)
net_layers = utility.get_child_modules(net)
layer_activations = [None] * len(net_layers)


def layer_hook_wrapper(idx):
    def hook(_module, inputs, outputs):
        layer_activations[idx] = inputs[0]
        # layer_activations[idx] = outputs
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
    # return [inputs] + layer_activations
    return layer_activations


# ======= Random Projections =======
RP = torch.randn((n_dims, n_random_projections), device=DEVICE)
RP = RP / RP.norm(2, dim=0)


def get_input(data): return data[0]


debug.expand_ignore = "DataLoader"


STD = not args.use_var
stats_path = os.path.join(MODELDIR, "stats_{}.pt")

stats_input_A = utility.collect_stats(
    DATA_A, get_input, n_classes, class_conditional=False,
    std=STD, path=stats_path.format('inputs'), device=DEVICE, use_drive=USE_DRIVE)
stats_input_A_C = utility.collect_stats(
    DATA_A, get_input, n_classes, class_conditional=True,
    std=STD, path=stats_path.format('inputs-CC'), device=DEVICE, use_drive=USE_DRIVE)

mean_A, std_A = stats_input_A
mean_A_C, std_A_C = stats_input_A_C

# # min_A, max_A = utility.collect_min_max(
# #     DATA_A, path=stats_path.format('min-max'), device=DEVICE, use_drive=USE_DRIVE)


# def project_RP(data):
#     X, Y = data
#     return (X - mean_A).reshape(-1, n_dims) @ RP


# def project_RP_CC(data):
#     X, Y = data
#     X_proj_C = None
#     for c in range(n_classes):
#         X_proj_c = (X[Y == c] - mean_A_C[c]).reshape(-1, n_dims) @ RP
#         if X_proj_C is None:
#             X_proj_C = torch.empty((X.shape[0], n_random_projections),
#                                    dtype=X_proj_c.dtype, device=X.device)
#         X_proj_C[Y == c] = X_proj_c
#     return X_proj_C


# # Random ReLU Projections
# relu_bias = (torch.randn((1, n_random_projections),
#                          device=DEVICE) * std_A.max())
# relu_bias_C = (torch.randn((n_classes, n_random_projections), device=DEVICE)
#                * std_A_C.max(dim=1, keepdims=True)[0].reshape(n_classes, 1))


# def project_RP_relu(data):
#     return F.relu(project_RP(data) + relu_bias)


# def project_RP_relu_CC(data):
#     X, Y = data
#     return F.relu(project_RP_CC(data) + relu_bias_C[Y])

# # ======= Combined =======


# def combine(project1, project2):
#     def _combined_fn(data):
#         out1 = project1(data)
#         out2 = project2(data)
#         if not isinstance(out1, list):
#             out1 = [out1]
#         if not isinstance(out2, list):
#             out2 = [out2]
#         return out1 + out2
#     return _combined_fn


# ======= Loss Function =======

STD = not args.use_var
stats_path = os.path.join(MODELDIR, "stats_{}.pt")


def regularization(x):
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    return (torch.norm(diff1) + torch.norm(diff2) +
            torch.norm(diff3) + torch.norm(diff4))


def loss_stats(stats_a, stats_b):
    if not isinstance(stats_a, list):
        stats_a, stats_b = [stats_a], [stats_b]
    assert len(stats_a) == len(stats_b), "lists need to be of same length"
    return sum(
        (ma - mb).norm() + (sa - sb).norm() if ma.ndim == 1 else
        (ma - mb).norm(dim=1).mean() +
        (sa - sb).norm(dim=1).mean()  # class_conditional
        for (ma, sa), (mb, sb) in zip(stats_a, stats_b))  # / len(stats_a)


# stats_A = [(m.running_mean, m.running_var.sqrt() if STD else m.running_var)
#            for m in net_layers]
# stats_A = utility.collect_stats(
#     DATA_A, project_NN_all, n_classes, class_conditional=False,
#     std=STD, path="models/stats_test.pt", device=DEVICE, use_drive=USE_DRIVE)
# stats_A = utility.collect_stats(
#     DATA_A, project_NN_all, n_classes, class_conditional=True,
#     std=STD, path="models/stats_test-CC.pt", device=DEVICE, use_drive=USE_DRIVE)

f_crit = args.f_crit
f_reg = args.f_reg
f_stats = args.f_stats
# f_input = args.f_input


# def loss_fn(data):
#     global net_last_outputs
#     net_last_outputs = None

#     inputs, labels = data
#     outputs = project_NN_all(data)

#     stats = utility.get_stats(
#         outputs, labels, n_classes, class_conditional=True, std=STD)

#     loss_obj = f_stats * loss_stats(stats, stats_A)

#     loss = loss_obj

#     if f_reg:
#         loss += f_reg * regularization(inputs)

#     if f_crit:
#         if net_last_outputs is None:
#             net_last_outputs = net(inputs)
#         loss_crit = f_crit * criterion(net_last_outputs, labels)
#         loss += loss_crit
#         info = {'loss_stats': loss_obj.item(),
#                 'loss_crit': loss_crit.item()}
#         return loss, info
#     return loss


def loss_fn_wrapper(name, project, class_conditional):
    _name = name.replace(' ', '-')
    stats_A = utility.collect_stats(
        DATA_A, project, n_classes, class_conditional,
        std=STD, path=stats_path.format(_name), device=DEVICE, use_drive=USE_DRIVE)

    # @debug
    def _loss_fn(data, project=project, class_conditional=class_conditional):
        global net_last_outputs
        net_last_outputs = None

        inputs, labels = data
        outputs = project(data)

        stats = utility.get_stats(
            outputs, labels, n_classes, class_conditional=class_conditional, std=STD)

        loss_obj = f_stats * loss_stats(stats, stats_A)

        loss = loss_obj

        loss += f_reg * regularization(inputs) if f_reg else 0

        # if f_input:
        #     stats_input = utility.get_stats(inputs, labels, n_classes,
        #                                     class_conditional, std=STD)
        #     loss += f_input * loss_stats(stats_input, stats_input_A)

        if f_crit:
            if net_last_outputs is None:
                net_last_outputs = net(inputs)
            loss += f_crit * criterion(net_last_outputs, labels)
            # info = {'loss_stats': loss_obj.item(),
            #         'loss_crit': loss_crit.item()}
            # return loss, info
        return loss
    return name, _loss_fn


methods = [
    # ("DI TEST", loss_fn)
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
        img_grid = torchvision.utils.make_grid(
            batch.cpu(), nrow=10, normalize=True, scale_each=True)
        plt.figure(figsize=(16, 32))
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


for method, loss_fn in methods:
    print("\n## Method:", method)

    batch = torch.randn((args.batch_size, *dataset.input_shape),
                        device=DEVICE, requires_grad=True)
    targets = torch.LongTensor(range(args.batch_size)).to(DEVICE) % n_classes
    DATA = [(batch, targets)]

    # print("Before:")
    # im_show(batch)

    def data_loss_fn(data):
        inputs, labels = data
        inputs = jitter(inputs)
        data = (inputs, labels)
        return loss_fn(data)

    optimizer = torch.optim.Adam([batch], lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    info = inversion.invert(DATA,
                            data_loss_fn,
                            optimizer,
                            #    scheduler=scheduler,
                            steps=inv_steps,
                            # steps=2,
                            # data_pre_fn=data_pre_fn,
                            # inputs_pre_fn=jitter,
                            #    track_history=True,
                            #    track_history_every=10,
                            plot=True,
                            use_amp=args.use_amp,
                            #    grad_norm_fn=grad_norm_fn,
                            )

    # ======= Result =======
    print("Inverted:")
    im_show(batch)
