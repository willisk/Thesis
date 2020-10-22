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

import ext.cifar10pretrained.cifar10_models.resnet as ext_resnet
importlib.reload(ext_resnet)

ResNet34 = ext_resnet.resnet34

LOGDIR = os.path.join(PWD, "exp01/runs")
shared.init_summary_writer(log_dir=LOGDIR)
# tb = shared.get_summary_writer("main")

# dataset = datasets.Dataset2D(type=3)
dataset = datasets.DatasetCifar10(
    load_dataset=True
)

# stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
# stats_net = dataset.load_statsnet(net=ResNet34(),
#                                   name="resnet34-pretrained",
#                                   resume_training=False,
#                                   use_drive=True,
#                                   )
# dataset.print_accuracy(stats_net)

print("================ INITED =======================")
# stats_net.disable_hooks()
# stats_net(torch.randn([5] + list(stats_net.input_shape)))
# stats_net.enable_hooks()


# plt.figure(figsize=(7, 7))
# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)

# # tb.add_figure("Data Set", plt.gcf(), close=False)
# plt.show()

# verify stats

# for h in stats_net.hooks.values():
#     m = h.modulex[0]
#     if isinstance(m, torch.nn.BatchNorm2d):
#         print("layer {}".format(h.name))
#         h_mean, h_var, h_cc = utility.reduce_mean_var(
#             h.running_mean, h.running_var, h.class_count)
#         utility.assert_mean_var(
#             # h.running_mean[0], h.running_var[0],
#             m.running_mean, m.running_var,
#             h_mean, h_var, h_cc)
#         print("layer {} asserted.".format(h.name))

# stats = stats_net.collect_stats()[0]
# mean = stats['running_mean']
# var = stats['running_var']

import torch.nn as nn


class CNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=None)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        return x

    def forward_verify(self, x):
        # print("verifying with batch size {}".format(len(x)))
        x = self.conv1(x)

        # labels = h.state().current_labels
        # shape = h.running_mean.shape
        # new_mean, new_var, m = utility.c_mean_var(x.detach(), labels, shape)

        batch_mean, batch_var = utility.batch_feature_mean_var(x)
        x = self.bn1(x)

        bn_mean, bn_var = self.bn1.running_mean, self.bn1.running_var

        # print("new_mean [1] calculated again, ", new_mean[1])
        # print("bn_mean  ", bn_mean)

        h = next(iter(self.bn1._forward_hooks.values())).__self__

        h_mean, h_var, h_cc = utility.reduce_mean_var(
            h.running_mean, h.running_var, h.class_count)

        # print("Assert true batch mean close to stats recorded mean")
        # utility.assert_mean_var(
        #     batch_mean, batch_var,
        #     h_mean, h_var, h_cc)

        utility.nan_to_zero_(h_mean)
        utility.nan_to_one_(h_var)

        # print("Assert bn.running_mean close to tracked and reduced mean")

        try:
            utility.assert_mean_var(
                bn_mean, bn_var,
                batch_mean, batch_var,
                # h_mean, h_var
            )
        except Exception as e:
            print(e)

        return x


net = CNet()
stats_net = statsnet.CStatsNet(net, 10)
# only interested in verifying by bn layer
net.conv1._forward_hooks.clear()
# X, Y = dataset.full()
# X, Y = next(iter(dataset.train_loader()))
stats_net.start_tracking_stats()
net.train()

# h = next(iter(stats_net.net.bn1._forward_hooks.values())).__self__
# h.reset()

for inputs, labels in dataset.train_loader():
    stats_net.current_labels = labels
    stats_net.net.forward_verify(inputs)


# meanx, varx = utility.batch_feature_mean_var(X)
# print("TOTAL true mean ", meanx)
# print("TOTAL true var ", varx)

# for c in range(dataset.get_num_classes()):
#     data = X[Y == c]
#     class_mean, class_var = utility.batch_feature_mean_var(data)
#     print("mean true: ", class_mean)
#     print("mean stored: ", mean[c])
#     assert np.allclose(mean[c], class_mean)
#     assert np.allclose(var[c], class_var)
#     print("class {} asserted.".format(c))
