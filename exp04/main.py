
import os
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)

import datasets
import statsnet
import utility
import deepinversion
import shared
import resnet_cifar10_nvlab

import importlib
importlib.reload(datasets)
importlib.reload(statsnet)
importlib.reload(utility)
importlib.reload(deepinversion)
importlib.reload(shared)
importlib.reload(resnet_cifar10_nvlab)

dataset = datasets.DatasetCifar10()
stats_net = dataset.load_statsnet(net=resnet_cifar10_nvlab.ResNet34(),
                                  name="cifar10-resnet34-nvlab",
                                  resume_training=True,
                                  use_drive=True
                                  )
