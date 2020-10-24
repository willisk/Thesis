"""Reconstruction Method Comparison
"""
import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

LOGDIR = os.path.join(PWD, "exp02/runs")
shared.init_summary_writer(LOGDIR)
tb = shared.get_summary_writer("main")

# matplotlib params
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['animation.html'] = 'jshtml'


def splot(nrows, ncols, i):
    ax = plt.subplot(nrows, ncols, i)
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    return ax


# # weight params
# weight_params = dict(
#     a=[0.001, 0.6, 1, 2, 1000],
#     b=[0.001, 0.6, 1, 2, 1000],
# )

# weight_params['b'] = list(reversed(weight_params['b']))

# nrows = len(weight_params['a'])
# ncols = len(weight_params['b'])

# w_param_product = utility.dict_product(weight_params)
# w_param_filtered = []

# # plt.figure(figsize=(7, 5))

# for i, w_param in enumerate(w_param_product):

#     ax = splot(nrows, ncols, i + 1)

#     if (utility.in_product_outer(nrows, ncols, i)
#             and not utility.in_product_edge(nrows, ncols, i)):
#         for spine in ax.spines.values():
#             spine.set_edgecolor('white')
#         continue
#     N = 11
#     weights = deepinversion.betabinom_distr(N=N, **w_param)
#     plt.bar(list(range(N)), weights)
#     w_param_filtered.append(w_param)


# xlabel = weight_params['b']
# ylabel = weight_params['a']
# xlabel[0] = "b = " + str(xlabel[0])
# ylabel[-1] = "a = " + str(ylabel[-1])
# utility.subplots_labels(xlabel, ylabel)

# tb.add_figure("Reconstruction Weights", plt.gcf(), close=False)
# plt.show()


# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)
# plt.show()


# UQ plots
dataset = datasets.Dataset2D(type=3)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
n_hooks = len(stats_net.hooks)


num_classes = dataset.get_num_classes()
# target_labels = torch.arange(num_classes) % num_classes


colors = ['Reds_r', 'Blues_r']


hyperparameters = dict(
    n_steps=[200],
    learning_rate=[0.1],
    factor_reg=[1],
    factor_input=[0.5],
    factor_layer=[0.5],
    factor_criterion=[1],
    distr_a=[1],
    distr_b=[1],
)

regularization = None


def perturbation(x):
    x.data += torch.randn_like(x) * 0.001
    return x


X, Y = dataset.X, dataset.Y

for hp in utility.dict_product(hyperparameters):

    comment = utility.dict_to_str(hp)
    print(comment)

    tb = shared.get_summary_writer(comment)
    plt.figure(figsize=(6, 3))

    # layer_weights = deepinversion.betabinom_distr(
    #     len(stats_net.hooks) - 1, hp['distr_a'], hp['distr_b'])

    # ncols = num_classes + 1
    ncols = num_classes

    # UQ plots
    # from kornia.losses.divergence import js_div_loss_2d as JSLoss
    # criterion = nn.CrossEntropyLoss(reduction='none')
    # def JSLoss(input, target):

    # criterion = JSLoss(reduction='none')

    for c in range(num_classes):

        # set up loss
        target_labels = torch.LongTensor([c])
        loss_fn = deepinversion.inversion_loss(stats_net, criterion, target_labels,
                                               regularization=regularization,
                                               reg_reduction_type='none',
                                               **hp)
        splot(1, ncols, c + 1)

        with torch.no_grad():
            utility.plot_contourf_data(
                X, loss_fn, n_grid=100, scale_grid=1.5, cmap=colors[c], levels=30,
                contour=True, colorbar=True)
        plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap='Spectral', alpha=.4)

    tb.add_figure("loss landscape", plt.gcf(), close=False)
    plt.show()

    # DeepInversion plots
    bs = 4
    target_labels = (torch.arange(bs)) % num_classes
    shape = [bs] + list(stats_net.input_shape)
    inputs = torch.randn(shape)

    optimizer = optim.Adam([inputs], lr=hp['learning_rate'])
    criterion = dataset.get_criterion()

    # set up loss
    loss_fn = deepinversion.inversion_loss(stats_net, criterion, target_labels,
                                           regularization=regularization,
                                           reg_reduction_type='sum',
                                           **hp)

    # with torch.no_grad():
    #     utility.plot_contourf_data(
    #         X, loss_fn, n_grid=100, scale_grid=1.5, cmap=colors[c], levels=30,
    #         contour=True, colorbar=True)
    invert = deepinversion.deep_inversion(inputs,
                                          stats_net,
                                          loss_fn,
                                          optimizer,
                                          steps=hp['n_steps'],
                                          #   perturbation=perturbation,
                                          #   projection=projection,
                                          track_history=True,
                                          track_history_every=20
                                          )

    print("inverted:")
    plt.figure(figsize=(6, 6))
    dataset.plot(stats_net)
    dataset.plot_history(invert, target_labels)

    tb.add_figure("loss landscape", plt.gcf(), close=False)
    plt.show()


# target_labels = torch.arange(num_classes) % num_classes
# criterion = nn.CrossEntropyLoss(reduction='sum')

# history = deepinversion.deep_inversion(stats_net,
#                                        criterion,
#                                        target_labels,
#                                        steps=200,
#                                        lr=0.1,
#                                        weights=w_selected,
#                                        #    track_history=False,
#                                        track_history=True,
#                                        track_history_every=10
#                                        )

# # plt.figure(figsize=(7, 7))
# dataset.plot(stats_net)
# dataset.plot_history(history, target_labels)

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

# tb.close()
