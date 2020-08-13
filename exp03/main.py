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

LOGDIR = os.path.join(PWD, "runs/exp03")
shared.init_summary_writer(log_dir=LOGDIR)
tb = shared.get_summary_writer("")

dataset = datasets.DatasetCifar10()


# stats_net = dataset.load_statsnet(resume_training=True, use_drive=True)
stats_net = dataset.load_statsnet(resume_training=False, use_drive=True)
# dataset.print_accuracy(stats_net)

# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)

# num_classes = dataset.get_num_classes()
# target_labels = torch.arange(num_classes) % num_classes
# history = deepinversion.deep_inversion(stats_net, dataset.get_criterion(),
#                                        target_labels,
#                                        steps=100,
#                                        lr=0.1,
#                                        track_history=False,
#                                        #    track_history=True,
#                                        #    track_history_every=10
#                                        )

# dataset.plot(stats_net)
# dataset.plot_history(history, target_labels)

# tb.add_figure("Data Reconstruction", plt.gcf(), close=False)
# plt.show()

# tb.close()
