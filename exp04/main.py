
import os
import sys

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import datasets
import statsnet
import utility
import deepinversion
import shared
from ext.cifar10pretrained.cifar10_models.resnet import resnet34 as ResNet34

import importlib
importlib.reload(datasets)
importlib.reload(statsnet)
importlib.reload(utility)
importlib.reload(deepinversion)
importlib.reload(shared)

dataset = datasets.DatasetCifar10()
stats_net = dataset.load_statsnet(net=ResNet34(),
                                  name="resnet34-pretrained",
                                  resume_training=False,
                                  use_drive=True
                                  )
