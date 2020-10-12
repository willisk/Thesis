import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from IPython.display import display, FileLink

import numpy as np

import torch
import torch.optim as optim
import torchvision.utils as vutils

import argparse

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

from ext.cifar10pretrained.cifar10_models.resnet import resnet34 as ResNet34

try:
    from apex import amp
    USE_AMP = True
except ImportError:
    print("Running without APEX.")
    USE_AMP = False

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

# matplotlib params
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['animation.html'] = 'jshtml'

# argparse
parser = argparse.ArgumentParser(description="DeepInversion on cifar-10")
parser.add_argument("-steps", "--n_steps", type=int, default=1000)
parser.add_argument("-bs", "--batch_size", type=int, default=32)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.05)
parser.add_argument("-fc", "--factor_criterion", type=float, default=1)
parser.add_argument("-fr", "--factor_reg", type=float, default=2.5e-5)
parser.add_argument("-fi", "--factor_input", type=float, default=0.0)
parser.add_argument("-fl", "--factor_layer", type=float, default=10)
parser.add_argument("-da", "--distr_a", type=float, default=1)
parser.add_argument("-db", "--distr_b", type=float, default=1)
parser.add_argument("--random_labels", action="store_true")
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--hp_sweep", action="store_true")
parser.add_argument("--track_history", action="store_true")
parser.add_argument("--track_history_every", type=int, default=10)
parser.add_argument("--use_drive", action="store_true")
parser.add_argument("--save_images", action="store_true")
parser.add_argument("-f", "--force", action="store_true")


if sys.argv[0] == 'ipykernel_launcher':
    args = parser.parse_args([])
    args.n_steps = 100
    args.perturb = True
    args.track_history = True
    args.save_images = True
    args.use_drive = True
    args.force = True
else:
    args = parser.parse_args()


# Folder setup
FIGDIR = os.path.join(PWD, "figures/cifar10-inversion")
LOGDIR = os.path.join(PWD, "exp03/runs")
if args.use_drive:
    FIGDIR, _ = utility.search_drive(FIGDIR)
    LOGDIR, _ = utility.search_drive(LOGDIR)

if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

# Tensorboard summary writer
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("main")


# set seed
np.random.seed(0)
torch.manual_seed(0)


dataset = datasets.DatasetCifar10(load_dataset=False)
# dataset = datasets.Dataset2D(type=3)

stats_net = dataset.load_statsnet(net=ResNet34(),
                                  name="resnet34-pretrained",
                                  resume_training=False,
                                  use_drive=args.use_drive
                                  )
stats_net.mask_bn_layer()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
stats_net.to(DEVICE)

if USE_AMP:
    data_type = torch.half
else:
    data_type = torch.float

# dataset.print_accuracy(stats_net)

# plot means
# dataset.plot_stats(stats_net)
# plt.show()


# set up deep inversion
# set up targets
criterion = dataset.get_criterion()


# hyperparameters

if args.hp_sweep:
    hyperparameters = dict(
        n_steps=[args.n_steps],
        batch_size=[32],
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
        batch_size=[args.batch_size],
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


for hp in utility.dict_product(hyperparameters):

    comment = utility.dict_to_str(hp)
    print("Hyperparameters:")
    print(comment)

    fig_path = os.path.join(FIGDIR, comment)
    tb = shared.get_summary_writer(comment)

    if args.hp_sweep and not any([hp['factor_input'], hp['factor_layer'], hp['factor_criterion'], hp['factor_reg']]):
        continue

    if not args.force and args.save_images and os.path.exists(fig_path + ".png"):
        continue

    # target_labels = (torch.arange(hp['batch_size']) %
    #                  dataset.get_num_classes()).to(DEVICE)
    target_labels = torch.LongTensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to(DEVICE)
    inputs = torch.randn([hp['batch_size']] + list(stats_net.input_shape),
                         requires_grad=True, device=DEVICE,
                         #  dtype=data_type
                         )
    optimizer = optim.Adam([inputs], lr=hp['learning_rate'])

    if USE_AMP:
        stats_net, optimizer = amp.initialize(
            stats_net, optimizer, opt_level='O1', loss_scale='dynamic')

    stats_net.eval()  # important, otherwise generated images will be non natural
    if USE_AMP:
        # need to do this trick for FP16 support of batchnorms
        stats_net.train()
        for module in stats_net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval().half()

    # set up loss
    loss_fn = deepinversion.inversion_loss(stats_net, criterion, target_labels,
                                           regularization=regularization,
                                           reg_reduction_type='mean',
                                           **hp)

    perturbation = jitter if args.perturb else None

    invert = deepinversion.deep_inversion(inputs,
                                          stats_net,
                                          loss_fn,
                                          optimizer,
                                          steps=hp['n_steps'],
                                          perturbation=perturbation,
                                          #   projection=projection,
                                          track_history=args.track_history,
                                          track_history_every=args.track_history_every,
                                          )

    # # dataset.plot(stats_net)
    target_labels = target_labels.cpu()
    for im, step in invert:
        vutils.save_image(im.data, fig_path + 'step={}'.format(step) + '.png',
                          normalize=True, scale_each=True, nrow=10)
    # frames = dataset.plot_history(invert, target_labels)

    # if len(frames) > 1:  # animated gif
    #     for _ in range(10):  # make last frame stick
    #         frames.append(frames[-1])
    #     anim = ArtistAnimation(plt.gcf(), frames,
    #                            interval=300, blit=True)
    #     plt.close()
    #     if args.save_images:
    #         anim.save(fig_path + ".gif", writer=PillowWriter())
    #         FileLink(fig_path + ".gif")
    #         dataset.plot_history([invert[-1]], target_labels)
    #         tb.add_figure("DeepInversion", plt.gcf(), close=False)
    #         plt.savefig(fig_path + ".png")
    #         plt.close()
    #     display(anim)
    # else:
    #     if args.save_images:
    #         tb.add_figure("DeepInversion", plt.gcf(), close=False)
    #         plt.savefig(fig_path + ".png")
    #     plt.show()

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

tb.close()
