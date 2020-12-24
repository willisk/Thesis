'''
ResNet model inversion for CIFAR10.
Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
This work is made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license, visit https://github.com/NVlabs/DeepInversion/blob/master/LICENSE
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import os
import sys
import glob
import collections


PWD = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PWD)
import utility
import importlib
import inversion
importlib.reload(utility)
# from resnet_cifar import ResNet34, ResNet18

from ext.cifar10pretrained.cifar10_models import resnet34 as ResNet34

# provide intermeiate information
debug_output = True


def jitter(x):
    off1, off2 = torch.randint(low=-2, high=2, size=(2, 1))
    x = torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    return x


def loss_fn(data):
    inputs, labels = data
    outputs = net(jitter(inputs))
    loss = 10 * sum(layer_losses)
    loss += 0.001 * regularization(inputs)
    loss += criterion(outputs, labels)
    return loss


def regularization(x):
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
    loss_var = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)
    return loss_var


if __name__ == "__main__":

    bs = 256
    iters = 400

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("loading resnet34")

    net = ResNet34()
    model_path = os.path.join(PWD, 'models/CIFAR10/net_resnet34.pt')
    save_path, load_path = utility.save_load_path(model_path, True)
    checkpoint = torch.load(load_path, map_location=device)

    net.to(device)
    net.load_state_dict(checkpoint)
    net.eval()

    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn((256, 3, 32, 32),
                         requires_grad=True, device=device)

    optimizer = optim.Adam([inputs], lr=0.1)

    batch_idx = 0
    prefix = "runs/data_generation/" + "try1" + "/"

    for create_folder in [prefix, prefix + "/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    print("Starting model inversion")

    criterion = nn.CrossEntropyLoss()

    targets = torch.LongTensor(range(bs)).to(device) % 10

    net_layers = utility.get_bn_layers(net)
    layer_losses = [None] * len(net_layers)

    def layer_hook_wrapper(idx):
        def hook(module, inputs, outputs):
            nch = inputs[0].shape[1]
            mean = inputs[0].mean([0, 2, 3])
            var = inputs[0].var([0, 2, 3])
            r_feature = ((mean - module.running_mean).norm() +
                         (var - module.running_var).norm())
            # var = inputs[0].permute(1, 0, 2, 3).contiguous().view(
            #     [nch, -1]).var(1, unbiased=False)
            # r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            #     module.running_mean.data.type(var.type()) - mean, 2)
            layer_losses[idx] = r_feature
        return hook

    for l, module in enumerate(net_layers):
        module.register_forward_hook(layer_hook_wrapper(l))

    info = inversion.invert([(inputs, targets)],
                            loss_fn,
                            optimizer,
                            #    scheduler=scheduler,
                            steps=iters,
                            # steps=2,
                            # data_pre_fn=data_pre_fn,
                            # inputs_pre_fn=jitter,
                            #    track_history=True,
                            #    track_history_every=10,
                            plot=True,
                            #    use_amp=args.use_amp,
                            #    grad_norm_fn=grad_norm_fn,
                            )

    import matplotlib.pyplot as plt

    def im_show(batch):
        with torch.no_grad():
            img_grid = torchvision.utils.make_grid(
                batch.cpu(), nrow=10, normalize=True, scale_each=True)
            plt.figure(figsize=(16, 32))
            plt.imshow(img_grid.permute(1, 2, 0))
            plt.show()

    im_show(inputs)
