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
stats_net = dataset.load_statsnet(net=ResNet34(),
                                  name="resnet34-pretrained",
                                  resume_training=False,
                                  use_drive=True,
                                  )
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


X, Y = dataset.full()
# X, Y = next(iter(dataset.train_loader()))
stats_net.start_tracking_stats()
# stats_net.reset()
stats_net.current_labels = Y[:50000]
stats_net.net.forward_verify(X[:50000])


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
