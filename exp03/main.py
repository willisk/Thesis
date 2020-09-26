import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
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

LOGDIR = os.path.join(PWD, "exp03/runs")
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("main")

plt.rcParams['figure.figsize'] = (6, 6)

dataset = datasets.DatasetCifar10(load_dataset=False)
# dataset = datasets.Dataset2D(type=3)

# stats_net = dataset.load_statsnet(resume_training=True, use_drive=True)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
# dataset.print_accuracy(stats_net)

# plot means
# dataset.plot_stats(stats_net)
# plt.show()


# set up deep inversion

# hyperparameters
learning_rate = 0.05

hyperparameters = dict(
    n_steps=[400],
    learning_rate=[0.05],
    criterion=[0, 1],
    input=[0, 0.0001, 0.01],
    regularization=[0, 0.0001, 0.01],
    layer=[0, 0.0001, 0.01],
    a=[1],
    b=[1],
)

# input_reg_factor = 0  # 1  # 0.0001
# layer_reg_factor = 0  # 1
# criterion_factor = 0
# reg_factor = 1  # 2.5e-5

# alpha = 1
# beta = 1


def projection(x):
    x.data.clamp_(-1, 1)
    return x


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
    tb = shared.get_summary_writer(comment)

    # hyperparameters
    n_steps = hp['n_steps']
    learning_rate = hp['learning_rate']
    input_reg_factor = hp['input']
    layer_reg_factor = hp['layer']
    criterion_factor = hp['criterion']
    reg_factor = hp['regularization']
    a = hp['a']
    b = hp['b']

    if not any([input_reg_factor, layer_reg_factor, criterion_factor, reg_factor]):
        continue

    layer_weights = deepinversion.betabinom_distr(
        len(stats_net.hooks) - 1, a, b)

    inputs = torch.randn(shape)
    optimizer = optim.Adam([inputs], lr=learning_rate)

    # set up loss
    def inversion_loss(x):
        stats_net.set_reg_reduction_type('mean')
        outputs = stats_net({'inputs': x, 'labels': target_labels})
        criterion_loss = criterion(outputs, target_labels)

        components = stats_net.get_hook_regularizations()
        input_reg = components.pop(0)
        layer_reg = sum([w * c for w, c in zip(layer_weights, components)])
        total_loss = (input_reg_factor * input_reg
                      + layer_reg_factor * layer_reg
                      + criterion_factor * criterion_loss
                      + reg_factor * regularization(x))
        return total_loss

    invert = deepinversion.deep_inversion(inputs,
                                          stats_net,
                                          inversion_loss,
                                          optimizer,
                                          steps=n_steps,
                                          #   track_history=True,
                                          perturbation=jitter,
                                          projection=projection,
                                          track_history_every=10
                                          )

    # # dataset.plot(stats_net)
    dataset.plot_history(invert, target_labels)

    if not os.path.exists("figures"):
        os.mkdir("figures")
    tb.add_figure("DeepInversion", plt.gcf(), close=False)
    plt.savefig("figures/inversion " + comment + ".png")
    plt.show()

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

tb.close()
