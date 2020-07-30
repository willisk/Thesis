import os
import sys
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(0)
torch.manual_seed(0)

import datasets
import statsnet
import utility
importlib.reload(datasets)
importlib.reload(statsnet)
importlib.reload(utility)

comment = ""
tb = SummaryWriter(comment=comment)


def deep_inversion(stats_net, criterion, labels, num_features=2, steps=5, track_history=False):

    stats_net.stop_tracking_stats()
    stats_net.enable_hooks()

    shape = [len(labels)] + list(stats_net.input_shape)
    inputs = torch.randn(shape, requires_grad=True)

    optimizer = optim.Adam([inputs], lr=0.1)

    if track_history:
        history = []
        history.append(inputs.data.detach().clone())

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = stats_net(data)
        loss = criterion(outputs, labels)

        # reg_loss = sum([s.regularization for s in net.stats])
        # reg_loss = stats_net.hooks[0].regularization

        # loss = loss + reg_loss
        loss = stats_net.hooks[0].regularization

        loss.backward()

        optimizer.step()

        if track_history:
            history.append(inputs.data.detach().clone())

    print("Finished Inverting")

    stats_net.disable_hooks()

    if track_history:
        return history
    return inputs.data.detach()


# dataset = datasets.Dataset2D(type=3)
dataset = datasets.DatasetDigits()
# dataset = datasets.DatasetIris()

stats_net = dataset.load_statsnet()
dataset.print_accuracy(stats_net)

num_classes = dataset.get_num_classes()
target_labels = torch.arange(num_classes) % num_classes
history = deep_inversion(stats_net, dataset.get_criterion(),
                         target_labels,
                         steps=100,
                         num_features=4,
                         track_history=False
                         #  track_history=True
                         )

dataset.plot(stats_net)
# dataset.plot_stats(stats_net)
dataset.plot_history(history, target_labels)

# # tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
plt.show()

tb.close()
