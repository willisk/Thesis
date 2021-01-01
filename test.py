import os
import sys

import random

import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision

import matplotlib.pyplot as plt

import numpy as np

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import inversion
import datasets
import debug
import nets

try:
    get_ipython()   # pylint: disable=undefined-variable
    interactive_notebook = True
except:
    interactive_notebook = False


if interactive_notebook:
    import importlib
    importlib.reload(utility)
    importlib.reload(inversion)
    importlib.reload(datasets)
    importlib.reload(debug)
    importlib.reload(nets)

from debug import debug

# dataset = datasets.CIFAR10()
dataset = datasets.MNIST()

MODELDIR = dataset.data_dir

A, B, B_val = dataset.get_datasets(size_A=-1, size_B=-1)

stats_path = os.path.join(MODELDIR, "stats_{}.pt")

stats_A = utility.collect_stats(
    None, None, None, None,
    std=False, path=stats_path.format('NN-ALL'), device='cpu', use_drive=True)

stats_A_C = utility.collect_stats(
    None, None, None, None,
    std=False, path=stats_path.format('NN-ALL-CC'), device='cpu', use_drive=True)

Y = A.dataset.dataset.train_labels
n = torch.zeros(10, dtype=torch.long)
for c in range(10):
    n[c] = (Y == c).sum().item()

r_stats = utility.reduce_stats(stats_A_C, n)

for (ma, sa), (mb, sb) in zip(stats_A, r_stats):
    print((ma - mb).norm(), (sa - sb).norm())

path, net = dataset.net()
stats_bn = [(m.running_mean, m.running_var)
            for m in utility.get_bn_layers(net)]
# debug(stats_bn)
# debug(stats_A)
print("\nBN")

for (ma, sa), (mb, sb) in zip(r_stats[1:], stats_bn):
    print((ma - mb).norm(), (sa - sb).norm())
