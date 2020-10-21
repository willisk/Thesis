"""Test if counting categorical mean and variance is implemented correctly
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

from ext.cifar10pretrained.cifar10_models.resnet import resnet34 as ResNet34

LOGDIR = os.path.join(PWD, "exp01/runs")
shared.init_summary_writer(log_dir=LOGDIR)
# tb = shared.get_summary_writer("main")

# dataset = datasets.Dataset2D(type=3)
dataset = datasets.DatasetCifar10()

# stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
stats_net = dataset.load_statsnet(net=ResNet34(),
                                  name="resnet34-pretrained",
                                  resume_training=False,
                                  use_drive=True,
                                  )
# dataset.print_accuracy(stats_net)

# plt.figure(figsize=(7, 7))
# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)

# # tb.add_figure("Data Set", plt.gcf(), close=False)
# plt.show()

# verify stats

stats = stats_net.collect_stats()[0]
mean = stats['running_mean']
var = stats['running_var']
print("mean shape ", mean.shape)
print("var shape ", var.shape)

X, Y = dataset.full()
for c in range(dataset.get_num_classes()):
    data = X[Y == c]
    class_mean = data.mean([0, 2, 3])
    class_var = data.permute(1, 0, 2, 3).contiguous().view(
        [data.shape[1], -1]).var(1, unbiased=False)
    print("mean stored: ", mean[c])
    print("mean true: ", class_mean[c])
    # print("class_mean shape ", class_mean.shape)
    # print("class_var shape ", class_var.shape)
    assert np.allclose(mean[c], class_mean)
    assert np.allclose(np.sqrt(var[c]), class_var)
    print("class {} asserted.".format(c))
# tb.close()
