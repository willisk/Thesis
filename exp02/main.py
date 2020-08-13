"""Reconstruction Method Comparison
"""
import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

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

LOGDIR = os.path.join(PWD, "runs/exp02")
shared.init_summary_writer(LOGDIR)
tb = shared.get_summary_writer("")


def splot(nrows, ncols, i):
    ax = plt.subplot(nrows, ncols, i)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    return ax


# weight params
weight_params = dict(
    a=[0.001, 0.6, 1, 2, 1000],
    b=[0.001, 0.6, 1, 2, 1000],
)

weight_params['b'] = list(reversed(weight_params['b']))

nrows = len(weight_params['a'])
ncols = len(weight_params['b'])

w_param_product = utility.dict_product(weight_params)
w_param_filtered = []

for i, w_param in enumerate(w_param_product):

    ax = splot(nrows, ncols, i + 1)

    if (utility.in_product_outer(nrows, ncols, i)
            and not utility.in_product_edge(nrows, ncols, i)):
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        continue
    N = 11
    weights = deepinversion.inversion_layer_weights(N=N, **w_param)
    plt.bar(list(range(N)), weights)
    w_param_filtered.append(w_param)


xlabel = weight_params['b']
ylabel = weight_params['a']
xlabel[0] = "b = " + str(xlabel[0])
ylabel[-1] = "a = " + str(ylabel[-1])
utility.subplots_labels(xlabel, ylabel)

tb.add_figure("Reconstruction Weights", plt.gcf(), close=False)
plt.show()

# UQ plots

dataset = datasets.Dataset2D(type=3)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
n_hooks = len(stats_net.hooks)


num_classes = dataset.get_num_classes()
# target_labels = torch.arange(num_classes) % num_classes


colors = ['Reds_r', 'Blues_r']
factor_params = [np.inf, 0, 0.5]

###
# w_param_filtered = [dict(a=0.001, b=1000)]
# factor_params = [0]
###
plt.figure(figsize=(7, 7))
dataset.plot(stats_net)
dataset.plot_stats(stats_net)
plt.show()

first = True
for w_param in w_param_filtered:

    for factor in factor_params:

        if not first and factor == np.inf:  # same plots
            continue
        first = False

        comment = "factor={} ".format(factor) + utility.dict_to_str(w_param)
        tb = shared.get_summary_writer(comment)
        plt.figure(figsize=(9, 3))

        weights = deepinversion.inversion_layer_weights(N=n_hooks, **w_param)
        w = deepinversion.inversion_loss_weights(weights, factor)

        ncols = num_classes + 1

        for c in range(num_classes):
            splot(1, ncols, c + 1)
            dataset.plot_uq(stats_net, c, w, cmap=colors[c])

        splot(1, ncols, ncols)
        x = list(range(n_hooks))
        plt.gca().set_ylim([0, 1])
        plt.bar(x, w[:-1])
        plt.bar(n_hooks, w[-1], color='red')
        plt.xticks(x)

        tb.add_figure("Loss Landscape", plt.gcf(), close=False)
        plt.show()
    break

w_param_selected = w_param_filtered[0]
weights = deepinversion.inversion_layer_weights(N=n_hooks, **w_param_selected)
w = deepinversion.inversion_loss_weights(weights, 0.5)

target_labels = torch.arange(num_classes) % num_classes
criterion = nn.CrossEntropyLoss(reduction='sum')

history = deepinversion.deep_inversion(stats_net,
                                       criterion,
                                       target_labels,
                                       steps=100,
                                       lr=0.1,
                                       weights=weights,
                                       #    track_history=False,
                                       track_history=True,
                                       track_history_every=10
                                       )

plt.figure(figsize=(7, 7))
dataset.plot(stats_net)
dataset.plot_history(history, target_labels)

tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
plt.show()

tb.close()
