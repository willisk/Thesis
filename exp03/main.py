import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from IPython.display import display, FileLink

import numpy as np
import torch
import torch.optim as optim
import argparse

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import datasets
import statsnet
import utility
import deepinversion
import shared

import importlib
importlib.reload(datasets)
importlib.reload(statsnet)
importlib.reload(utility)
importlib.reload(deepinversion)
importlib.reload(shared)

# Folder setup
FIGDIR, _ = utility.search_drive(
    os.path.join(PWD, "figures/cifar10-inversion"))
if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

LOGDIR, _ = utility.search_drive(os.path.join(PWD, "exp03/runs"))
# Tensorboard summary writer
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("main")

# matplotlib params
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['animation.html'] = 'jshtml'

# argparse
parser = argparse.ArgumentParser(description="DeepInversion on cifar-10")
parser.add_argument("-steps", "--n_steps", type=int, default=100)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
parser.add_argument("-fc", "--factor_criterion", type=float, default=1)
parser.add_argument("-fr", "--factor_reg", type=float, default=0.0001)
parser.add_argument("-fi", "--factor_input", type=float, default=0.0001)
parser.add_argument("-fl", "--factor_layer", type=float, default=0.001)
parser.add_argument("-da", "--distr_a", type=float, default=1)
parser.add_argument("-db", "--distr_b", type=float, default=1)
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--hp_sweep", action="store_true")
parser.add_argument("--track_history", action="store_true")
parser.add_argument("--track_history_every", type=int, default=10)

if sys.argv[0] == 'ipykernel_launcher':
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# set seed
np.random.seed(0)
torch.manual_seed(0)


dataset = datasets.DatasetCifar10(load_dataset=False)
# dataset = datasets.Dataset2D(type=3)

stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
# dataset.print_accuracy(stats_net)

# plot means
# dataset.plot_stats(stats_net)
# plt.show()


# set up deep inversion

# hyperparameters

if args.hp_sweep:
    hyperparameters = dict(
        n_steps=[args.n_steps],
        learning_rate=[0.01],
        factor_criterion=[1, 0],
        factor_reg=[0.001, 0.0001, 0],
        factor_input=[0.001, 0.0001, 0],
        factor_layer=[0.01, 0.001, 0.0001, 0],
        distr_a=[1],
        distr_b=[1],
    )
else:
    hyperparameters = dict(
        n_steps=[args.n_steps],
        learning_rate=[args.learning_rate],
        factor_reg=[args.factor_reg],
        factor_input=[args.factor_input],
        factor_layer=[args.factor_layer],
        factor_criterion=[args.factor_criterion],
        distr_a=[args.distr_a],
        distr_b=[args.distr_b],
    )


def projection(x):
    x.data.clamp_(-1, 1)
    return x


def jitter(x):
    off1, off2 = torch.randint(low=-2, high=2, size=(2, 1))
    x.data = torch.roll(x.data, shifts=(off1, off2), dims=(2, 3))
    return x


def regularization(x):
    # apply total variation regularization
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    loss_var = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)

    # loss_l2 = torch.norm(x, 2)

    return loss_var


# set up targets
num_classes = dataset.get_num_classes()
target_labels = (torch.arange(6)) % num_classes
shape = [len(target_labels)] + list(stats_net.input_shape)
criterion = dataset.get_criterion()


for hp in utility.dict_product(hyperparameters):

    comment = utility.dict_to_str(hp)
    print(comment)

    fig_path = os.path.join(FIGDIR, comment)
    tb = shared.get_summary_writer(comment)

    if not any([hp['factor_input'], hp['factor_layer'], hp['factor_criterion'], hp['factor_reg']]):
        continue

    if os.path.exists(fig_path + ".png"):
        continue

    inputs = torch.randn(shape)
    optimizer = optim.Adam([inputs], lr=hp['learning_rate'])

    layer_weights = deepinversion.betabinom_distr(
        len(stats_net.hooks) - 1, hp['distr_a'], hp['distr_b'])

    # set up loss
    def inversion_loss(x):
        stats_net.set_reg_reduction_type('mean')
        outputs = stats_net({'inputs': x, 'labels': target_labels})
        criterion_loss = criterion(outputs, target_labels)

        components = stats_net.get_hook_regularizations()
        input_reg = components.pop(0)
        layer_reg = sum([w * c for w, c in zip(layer_weights, components)])
        total_loss = (hp['factor_input'] * input_reg
                      + hp['factor_layer'] * layer_reg
                      + hp['factor_criterion'] * criterion_loss
                      + hp['factor_reg'] * regularization(x))
        return total_loss

    if args.perturb:
        perturbation = jitter
    else:
        perturbation = None

    invert = deepinversion.deep_inversion(inputs,
                                          stats_net,
                                          inversion_loss,
                                          optimizer,
                                          steps=hp['n_steps'],
                                          perturbation=perturbation,
                                          projection=projection,
                                          track_history=args.track_history,
                                          track_history_every=args.track_history_every
                                          )

    # # dataset.plot(stats_net)
    print("inverted:")
    frames = dataset.plot_history(invert, target_labels)

    if len(frames) > 1:  # animated gif
        anim = ArtistAnimation(plt.gcf(), frames,
                               interval=300, repeat_delay=8000, blit=True)
        plt.close()
        anim.save(fig_path + ".gif", writer=PillowWriter())
        display(anim)
        FileLink(fig_path + ".gif")
        dataset.plot_history([invert[-1]], target_labels)
        tb.add_figure("DeepInversion", plt.gcf(), close=False)
        plt.savefig(fig_path + ".png")
        plt.close()
    else:
        tb.add_figure("DeepInversion", plt.gcf(), close=False)
        plt.savefig(fig_path + ".png")
        plt.show()

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

tb.close()
