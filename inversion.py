"""Testing reconstruction methods"""
import os
import sys

import argparse
from collections import defaultdict

import torch
# from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
# plt.style.use('default')

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)

USE_DRIVE = True

from utils import utility
from utils import methods
from utils import datasets
from utils import debug
from utils import nets

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
parser.add_argument("--reset_stats", action="store_true")
parser.add_argument("--save_run", action="store_true")
parser.add_argument("-methods", nargs='+', type=str)

if 'ipykernel_launcher' in sys.argv[0]:
    # args = parser.parse_args('-dataset MNIST'.split())
    args = parser.parse_args('-dataset CIFAR10'.split())
    args.inv_steps = 2
    args.size_A = 16
    args.size_B = 8
    args.batch_size = 8
    # args.save_run = True
else:
    args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
utility.seed_everything(args.seed)


print("#", __doc__)
print("# on", args.dataset)


# ======= Hyperparameters =======
print("Hyperparameters:")
print(utility.dict_to_str(vars(args), '\n'), '\n')
print(f"Running on '{DEVICE}'\n")


# ======= Create Dataset =======

if args.dataset == 'CIFAR10':
    dataset = datasets.CIFAR10()
elif args.dataset == 'MNIST':
    dataset = datasets.MNIST()

A, B, _ = dataset.get_datasets(size_A=args.size_A, size_B=args.size_B)


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
              resume=nn_resume_training,
              reset=nn_reset_training,
              plot=True,
              use_drive=USE_DRIVE,
              )
net.eval()

if not 'ipykernel_launcher' in sys.argv[0]:
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
    if not 'ipykernel_launcher' in sys.argv[0]:
        print("verification ", end='')
        utility.print_net_accuracy(
            verification_net, DATA_A, estimate_epochs=10)
print()


def fig_path_fmt(*name_args, filetype="png"):
    if args.save_run:
        path = "figures"
        path = f"{path}/inversion_{args.dataset}_{'_'.join(name_args)}.{filetype}".replace(
            ' ', '_')
        save_path, _ = utility.search_drive(path, use_drive=USE_DRIVE)
        return save_path
    return None


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
        continue

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
            info['reference'] = ideal_value
        return info

    def callback_fn(epoch, metrics):
        if args.show_after > 0 and epoch % args.show_after == 0:
            print(f"\nepoch {epoch}:", flush=True)
            utility.im_show(batch[:10],
                            fig_path_fmt(f"{method}_epoch_{epoch}"),
                            scale_each=True)

    optimizer = torch.optim.Adam([batch], lr=inv_lr)
    # scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    metrics_fig_path = fig_path_fmt(method, "pdf")

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
    print("Inverted:")
    utility.im_show(batch, fig_path_fmt(
        f"{method}_epoch_{inv_steps}_full"),
        scale_each=True)

    accuracy = utility.net_accuracy(net, DATA_B)
    print(f"\tnn accuracy: {accuracy * 100:.1f} %")

    metrics[method]['acc'] = accuracy

    if verification_net:
        accuracy_ver = utility.net_accuracy(verification_net, DATA_B)
        print(f"\tnn verification accuracy: {accuracy_ver * 100:.1f} %")
        metrics[method]['acc(ver)'] = accuracy_ver

print("\n# Summary")
print("=========\n")


table_path = fig_path_fmt(f"results", "csv")
utility.make_table(metrics, row_name="method", out=table_path)


def plot_metrics(method, **kwargs):
    print(f"\n## {method}")
    utility.plot_metrics(plots[method], **kwargs)
