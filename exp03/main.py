import os
import sys
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(0)
torch.manual_seed(0)

import datasets
import statsnet
import utility
import deepinversion

import importlib
importlib.reload(datasets)
importlib.reload(statsnet)
importlib.reload(utility)
importlib.reload(deepinversion)

comment = ""
tb = SummaryWriter(comment=comment)


# dataset = datasets.Dataset2D(type=3)
# dataset = datasets.DatasetDigits()
# dataset = datasets.DatasetIris()
dataset = datasets.DatasetImagenet()


stats_net = dataset.load_statsnet()
dataset.print_accuracy(stats_net)

num_classes = dataset.get_num_classes()
target_labels = torch.arange(num_classes) % num_classes
# history = deepinversion.deep_inversion(stats_net, dataset.get_criterion(),
#                                        target_labels,
#                                        steps=100,
#                                        track_history=False
#                                        #  track_history=True
#                                        )

# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)
# dataset.plot_history(history, target_labels)

# # tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
plt.show()

tb.close()
