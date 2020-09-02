import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

np.random.seed(0)
torch.manual_seed(0)


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

LOGDIR = os.path.join(PWD, "exp03/runs")
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("main")

dataset = datasets.DatasetCifar10()

# stats_net = dataset.load_statsnet(resume_training=True, use_drive=True)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
# dataset.print_accuracy(stats_net)

# dataset.plot(stats_net)
dataset.plot_stats(stats_net)
plt.show()

num_classes = dataset.get_num_classes()
target_labels = torch.arange(num_classes) % num_classes

weights = deepinversion.inversion_layer_weights(N=len(stats_net.hooks))
weights = deepinversion.inversion_loss_weights(weights, 1)


def projection(x):
    return torch.clamp(x, -1, 1)


def jitter(x):
    off1, off2 = torch.randint(low=-2, high=2, size=(2, 1))
    return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


def regularization(x):
    # apply total variation regularization
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    loss_var = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)

    loss_l2 = torch.norm(x, 2)

    return 2.5e-5 * loss_var + 0.0001 * loss_l2


invert = deepinversion.deep_inversion(stats_net, dataset.get_criterion(),
                                      target_labels,
                                      steps=200,
                                      lr=0.05,
                                      track_history=False,
                                      weights=weights,
                                      perturbation=jitter,
                                      regularization=regularization,
                                      projection=projection,
                                      #    track_history=True,
                                      #    track_history_every=10
                                      )

# dataset.plot(stats_net)
dataset.plot_history(invert, target_labels)
plt.show()

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

# tb.close()
