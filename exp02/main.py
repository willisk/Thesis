"""Reconstruction Method Comparison
"""
import os
import sys

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

import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

LOGDIR = os.path.join(PWD, "exp02/runs")
shared.init_summary_writer(LOGDIR)
writer = shared.get_summary_writer("main")


dataset = datasets.Dataset2D(type=3)
# dataset = datasets.DatasetGMM(
#     n_dims=2, n_classes=3, n_modes=8,
#     scale_mean=5, scale_cov=1,
#     n_samples_per_class=int(1e3)
# )
# print(dataset.JS())

stats_net = dataset.load_statsnet(
    resume_training=False,
    use_drive=True,
)
# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)
# plt.show()

num_classes = dataset.get_num_classes()
criterion = dataset.get_criterion(reduction='none')


hyperparameters = dict(
    method=['standard'],
    cc=[True],
    mask_bn=[False],
    use_bn_stats=[False],
    n_steps=[200],
    learning_rate=[0.1],
    factor_reg=[0],
    factor_input=[0],
    factor_layer=[0],
    factor_criterion=[1],
    distr_a=[1],
    distr_b=[1],
)

regularization = None


def perturbation(x, noise_level=0.001):
    x.data += torch.randn_like(x) * noise_level
    return x


cmaps = utility.categorical_cmaps(num_classes)
for hp in utility.dict_product(hyperparameters):
    comment = utility.dict_to_str(hp)
    print("Hyperparameters:")
    print(comment)

    # layer_len = len(stats_net.hooks) - 1
    # for l in range(layer_len):
    #     print(f"Regularization layer {l+1}")
    #     layer_weights = [0] * layer_len
    #     layer_weights[l] = 1

    #     # writer = shared.get_summary_writer(comment)
    #     plt.figure(figsize=(6, 3))

    #     # layer_weights = deepinversion.betabinom_distr(
    #     #     len(stats_net.hooks) - 1, hp['distr_a'], hp['distr_b'])

    #     for c in range(num_classes):
    #         target = torch.LongTensor([c])
    #         loss_fn = deepinversion.inversion_loss(stats_net, criterion, target, hp,
    #                                                #   regularization=regularization,
    #                                                layer_weights=layer_weights,
    #                                                reg_reduction_type='none',
    #                                                )
    #         plt.subplot(1, num_classes, c + 1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.tight_layout()
    #         dataset.plot(loss_fn=loss_fn, cmap=cmaps[c])
    #     plt.show()

    # writer.add_figure("loss landscape", plt.gcf(), close=False)

    # DeepInversion plots
    bs = 8
    targets = (torch.arange(bs)) % num_classes
    shape = [bs] + list(stats_net.input_shape)
    inputs = torch.rand(shape) * 20 - 10

    optimizer = torch.optim.Adam([inputs], lr=hp['learning_rate'])
    criterion = dataset.get_criterion()

    # set up loss
    loss_fn = deepinversion.inversion_loss(stats_net, criterion, targets, hp
                                           #    regularization=regularization,
                                           )

    utility.print_accuracy(inputs, targets, stats_net)
    invert = deepinversion.deep_inversion(inputs,
                                          stats_net,
                                          loss_fn,
                                          optimizer,
                                          steps=hp['n_steps'],
                                          #   pre_fn=perturbation,
                                          track_history=True,
                                          track_history_every=10
                                          )

    print("inverted:")
    plt.figure(figsize=(6, 6))
    dataset.plot(stats_net)
    dataset.plot_history(invert, targets)
    utility.print_accuracy(inputs, targets, stats_net)

    writer.add_figure("loss landscape", plt.gcf(), close=False)
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

#     ax = plt.subplot(nrows, ncols, i + 1)
#     plt.xticks([])
#     plt.yticks([])

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
