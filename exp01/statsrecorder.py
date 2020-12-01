# import numpy as np
import os
import sys
import torch

import argparse

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import utility
import importlib
importlib.reload(utility)

# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

# pylint: disable=no-member
torch.manual_seed(0)


n_classes = 3
n_features = 5
n_runs = 10
assert_on = False
dtype = torch.double
cap_gamma = 1

print(f"{dtype}")
print(f"cap gamma {cap_gamma}")


class StatsRecorder:
    def __init__(self, n_classes, data=None, labels=None):
        self.n_classes = n_classes
        self.initialized = False

        if data is not None:
            n_features = data.shape[1]
            shape = (n_classes, n_features)

            self.mean = torch.zeros(shape, dtype=data.dtype)
            self.var = torch.zeros(shape, dtype=data.dtype)
            self.mean, self.var, self.n = utility.c_stats(
                data, labels, n_classes, return_count=True)

            self.initialized = True

    def update(self, data, labels):
        if not self.initialized:
            self.__init__(self.n_classes, data, labels)
        else:
            new_mean, new_var, m = utility.c_stats(
                data, labels, self.n_classes, return_count=True)
            old_mean, old_var, n = self.mean, self.var, self.n

            self.mean, self.var, self.n = utility.combine_mean_var(old_mean, old_var, n,
                                                                   new_mean, new_var, m,
                                                                   cap_gamma=cap_gamma)


stats = StatsRecorder(n_classes)

data_shape = [n_features, 32, 32]
data = [torch.empty([0] + data_shape)] * n_classes

# assert_on = True

import torch.nn as nn
bn_layer = nn.BatchNorm2d(n_features)
bn_layer.train()

for i in range(n_runs):
    n_samples = torch.randint(10, 101, size=(1,)).item()
    new_data = torch.randn((n_samples, *data_shape))
    new_labels = torch.randint(n_classes, size=(n_samples,))

    bn_layer(new_data)
    new_data = new_data.to(dtype)

    # print("incoming data shape: ", new_data.shape)

    for c in range(n_classes):
        data[c] = torch.cat((data[c], new_data[new_labels == c]))
        # print("data[{}].shape: ".format(c), data[c].shape)

    stats.update(new_data, new_labels)

    for c in range(n_classes):
        # data: [n_class] * [n_cc_batch, n_feature, 32, 32]
        # print(data[0].shape)
        class_mean, class_var = utility.batch_feature_stats(data[c])

        # utility.assert_mean_var(class_mean, class_var,
        #                         stats.mean[c], stats.var[c], stats.n)

        if assert_on:
            s_mean, s_var = stats.mean.T, stats.var.T
            class_mean, class_var = class_mean.T, class_var.T
            assert s_mean[stats.n.flatten() != 0].isfinite().all(), \
                "recorded mean has invalid entries"
            assert s_mean[c].shape == class_mean.shape
            assert torch.allclose(s_mean[c], class_mean, equal_nan=True), \
                "class {}".format(c) \
                + "\nclass mean: {}".format(class_mean) \
                + "\nrecorded mean: {}".format(s_mean[c]) \
                + "\nerror: {}".format(torch.norm(s_mean[c] - class_mean))

            assert s_var[c].shape == class_var.shape
            assert s_var[stats.n.flatten() != 0].isfinite().all(), \
                "recorded var has invalid entries"
            assert torch.allclose(s_var[c], class_var, equal_nan=True), \
                "class {}".format(c) \
                + "\nclass var: {}".format(class_var) \
                + "\nrecorded var: {}".format(s_var[c]) \
                + "\nerror: {}".format(torch.norm(s_var[c] - class_var))

    # if i % 10 == 0:
    print("assert {} passed".format(i))


mean, var = torch.empty_like(stats.mean.T), torch.empty_like(stats.mean.T)
mean, var = mean.to(torch.double), var.to(torch.double)
for c in range(n_classes):
    data[c] = data[c].to(torch.double)
    m, v = utility.batch_feature_stats(data[c])
    mean[c] = m.T
    var[c] = v.T

mean, var = mean.T, var.T

data = torch.cat(data)
print("n_samples", len(data))
# stats.mean, stats.var = stats.mean.to(torch.float), stats.var.to(torch.float)
print("cond mean error: ", torch.norm(stats.mean - mean).item())
print("cond var error: ", torch.norm(stats.var - var).item())

mean, var, _ = utility.reduce_mean_var(stats.mean, stats.var, stats.n)
true_mean, true_var = utility.batch_feature_stats(data)
print("reduced mean error: ", torch.norm(mean - true_mean).item())
print("reduced var error: ", torch.norm(var - true_var).item())

print("bn mean error: ", torch.norm(bn_layer.running_mean - true_mean).item())
print("bn var error: ", torch.norm(bn_layer.running_var - true_var).item())
