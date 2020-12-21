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

USE_APEX = False

PWD = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PWD)
import utility
import importlib
importlib.reload(utility)
# from resnet_cifar import ResNet34, ResNet18

from ext.cifar10pretrained.cifar10_models import resnet34 as ResNet34

# provide intermeiate information
debug_output = False
debug_output = True


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view(
            [nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


def jitter(x):
    off1, off2 = torch.randint(low=-2, high=2, size=(2, 1))
    x = torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    return x


def loss_fn(data):
    inputs, labels = data
    outputs = net(inputs)
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
    net.eval()

    criterion = nn.CrossEntropyLoss()

    # place holder for inputs
    inputs = torch.randn((256, 3, 32, 32),
                         requires_grad=True, device=device)

    optimizer = optim.Adam([inputs], lr=0.1)

    net.load_state_dict(checkpoint)

    batch_idx = 0
    prefix = "runs/data_generation/" + "try1" + "/"

    for create_folder in [prefix, prefix + "/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    print("Starting model inversion")

    best_cost = 1e6

    criterion = nn.CrossEntropyLoss()

    optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

    targets = torch.LongTensor(range(bs)).to(device) % 10

    net_layers = utility.get_bn_layers(net)
    # layer_activations = [None] * len(net_layers)
    layer_losses = [None] * len(net_layers)

    def layer_hook_wrapper(idx):
        def hook(module, inputs, outputs):
            nch = inputs[0].shape[1]
            mean = inputs[0].mean([0, 2, 3])
            var = inputs[0].permute(1, 0, 2, 3).contiguous().view(
                [nch, -1]).var(1, unbiased=False)
            r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
                module.running_mean.data.type(var.type()) - mean, 2)
            layer_losses[idx] = r_feature
        return hook

        # Create hooks for feature statistics catching
    l = 0
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.register_forward_hook(layer_hook_wrapper(l))
            l += 1
    # for l, layer in enumerate(net_layers):
    #     layer.register_forward_hook(layer_hook_wrapper(l))

    for epoch in range(iters):
        inputs_jit = jitter(inputs)

        optimizer.zero_grad()
        loss = loss_fn((inputs, targets))

        if debug_output and epoch % 200 == 0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f}")
            vutils.save_image(inputs.data.clone(),
                              './{}/output_{}.png'.format(prefix,
                                                          epoch // 200),
                              normalize=True, scale_each=True, nrow=10)

        if best_cost > loss.item():
            best_cost = loss.item()
            best_inputs = inputs.data

        # backward pass
        loss.backward()

        optimizer.step()

    outputs = net(best_inputs)
    _, predicted_teach = outputs.max(1)

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      './{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each=True, nrow=10)
