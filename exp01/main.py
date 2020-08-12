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

LOGDIR = os.path.join(PWD, "runs/exp01")
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("")

dataset = datasets.Dataset2D(type=3)

stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
dataset.print_accuracy(stats_net)

plt.figure(figsize=(7, 7))
dataset.plot(stats_net)
dataset.plot_stats(stats_net)

tb.add_figure("Data Set", plt.gcf(), close=False)
plt.show()

# verify stats

stats = stats_net.collect_stats()[0]
mean = stats['running_mean']
var = stats['running_var']

X, Y = dataset.full()
for c in range(dataset.get_num_classes()):
    data = X[Y == c]
    assert np.allclose(mean[c], data.mean(axis=0))
    assert np.allclose(np.sqrt(var[c]), data.std(axis=0))
tb.close()
