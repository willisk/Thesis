"""Reconstruction Method Comparison
"""
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

LOGDIR = os.path.join(PWD, "runs/exp02")
shared.init_summary_writer(LOGDIR)

# weight params
weight_params = dict(
    a=[0.001, 0.6, 1, 2, 1000],
    b=[0.001, 0.6, 1, 2, 1000],
)

nrows = len(weight_params['a'])
ncols = len(weight_params['b'])

w_param_product = utility.dict_product(weight_params)
w_param_filtered = []

n_hooks = 11

for i, w_param in enumerate(w_param_product):

    ax = plt.subplot(nrows, ncols, i + 1)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])

    if (utility.in_product_outer(nrows, ncols, i)
            and not utility.in_product_edge(nrows, ncols, i)):
        plt.plot()
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        w_param_filtered.append(None)
        continue
    w_param_filtered.append(w_param)
    continue
    weights = deepinversion.inversion_layer_weights(N=n_hooks, **w_param)
    plt.bar(list(range(n_hooks)), weights)


xlabel = weight_params['a']
ylabel = weight_params['b']
xlabel[0] = "a = " + str(xlabel[0])
ylabel[-1] = "b = " + str(xlabel[-1])
utility.subplots_labels(xlabel, ylabel)

plt.show()

# UQ plots

dataset = datasets.Dataset2D(type=3)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
n_hooks = len(stats_net.hooks)


num_classes = dataset.get_num_classes()
# target_labels = torch.arange(num_classes) % num_classes


for w_param in w_param_filtered:

    # ax = plt.subplot(nrows, ncols, i + 1)

    weights = deepinversion.inversion_layer_weights(N=n_hooks, **w_param)
    comment = utility.dict_to_str(w_param)
    print(comment)
    print("hooks: ", n_hooks)
    print("weights: ", len(weights))
    # tb = shared.get_summary_writer(comment)
    plt.figure(figsize=(9, 6))
    # plt.title("class={} ".format(c) + comment)
    colors = ['Reds_r', 'Blues_r']
    for c in range(num_classes):
        plt.subplot(2, 3, c * 3 + 3)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        print(weights)
        print(w[:-1])
        plt.bar(list(range(n_hooks)), weights)
        plt.bar(n_hooks, w[-1], color='red')

        plt.subplot(2, 3, c * 3 + 1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        w = deepinversion.inversion_loss_weights(weights, 1)
        print("w: ", len(w[:-1]))
        dataset.plot_uq(stats_net, w, c, cmap=colors[c])
        plt.subplot(2, 3, c * 3 + 2)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        w = deepinversion.inversion_loss_weights(weights, 2)
        dataset.plot_uq(stats_net, w, c, cmap=colors[c])

    plt.show()
    break


# tb.close()
