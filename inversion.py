"""Testing reconstruction methods"""
import os
import sys

import argparse
from collections import defaultdict

import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision

import matplotlib.pyplot as plt
# plt.style.use('default')

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

USE_DRIVE = True

import utils.utility as utility
import utils.methods as methods
import utils.datasets as datasets
import utils.debug as debug
import utils.nets as nets

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
parser.add_argument("--nn_resume_train", action="store_true")
parser.add_argument("--nn_reset_train", action="store_true")
parser.add_argument("--use_amp", action="store_true")
parser.add_argument("--use_std", action="store_true")
parser.add_argument("--use_jitter", action="store_true")
parser.add_argument("--plot_ideal", action="store_true")
parser.add_argument("--scale_each", action="store_true")
parser.add_argument("--reset_stats", action="store_true")

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset MNIST'.split())
    args = parser.parse_args('-dataset CIFAR10'.split())
    args.inv_steps = 2
    args.size_A = 16
    args.size_B = 8
    args.batch_size = 8
else:
    args = parser.parse_args()

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
if args.plot_ideal:
    test_data = utility.DataL(
        B, batch_size=-1, shuffle=True, device=DEVICE)
    ideal_data = next(iter(test_data))

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

    batch = torch.randn((args.size_B, *input_shape),
                        requires_grad=True, device=DEVICE)
    targets = torch.LongTensor(
        range(args.size_B)).to(DEVICE) % n_classes
    DATA_B = [(batch, targets)]

    ideal_value = None

    def data_loss_fn(data):
        inputs, labels = data
        if args.use_jitter:
            inputs = jitter(inputs)
            data = (inputs, labels)
        info = loss_fn(data)
        if args.plot_ideal:
            global ideal_value
            if ideal_value is None:
                with torch.no_grad():
                    ideal_value = loss_fn(ideal_data)['loss'].item()
            info[':--: ideal'] = ideal_value
        return info

    def callback_fn(epoch, metrics):
        if epoch % args.show_after == 0:
            print(f"\nepoch {epoch}:", flush=True)
            im_show(batch[:10])

    optimizer = torch.optim.Adam([batch], lr=inv_lr)
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
                          )
    plots[method] = info

    # ======= Result =======
    print("Inverted:")
    im_show(batch)

    accuracy = utility.net_accuracy(net, DATA_B)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")

    metrics[method]['acc'] = accuracy

    if verifier_net:
        accuracy_ver = utility.net_accuracy(verifier_net, DATA_B)
        print(f"\tnn verifier accuracy: {accuracy_ver * 100:.1f} %")
        metrics[method]['acc(ver)'] = accuracy_ver

print("\n# Summary")
print("=========\n")

utility.make_table(
    metrics,
    row_name="method",
    out="figures/table_inversion_results.csv")


def plot_metrics(method, **kwargs):
    print(f"\n## {method}")
    utility.plot_metrics(plots[method], **kwargs)
