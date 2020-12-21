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

# from resnet_cifar import ResNet34, ResNet18
from ext.cifar10pretrained.cifar10_models import resnet34 as ResNet34

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    USE_APEX = True
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    print("will attempt to run without it")
    USE_APEX = False

PWD = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PWD)
import utility
import importlib
importlib.reload(utility)

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


def get_images(net, bs=256, epochs=1000, idx=-1, var_scale=0.00005,
               prefix=None, competitive_scale=0.01, global_iteration=None,
               optimizer=None, inputs=None, bn_reg_scale=0.0, random_labels=False,
               device='cpu'):

    best_cost = 1e6

    inputs.data = torch.randn(
        (bs, 3, 32, 32), requires_grad=True, device=device)

    criterion = nn.CrossEntropyLoss()

    optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

    targets = torch.LongTensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to(device)

    # Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # setting up the range for jitter
    lim_0, lim_1 = 2, 2

    for epoch in range(epochs):
        # apply random jitter offsets
        off1 = random.randint(-lim_0, lim_0)
        off2 = random.randint(-lim_1, lim_1)
        inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

        # foward with jit images
        optimizer.zero_grad()
        outputs = net(inputs_jit)
        loss = criterion(outputs, targets)
        loss_target = loss.item()

        # apply total variation regularization
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var = torch.norm(diff1) + torch.norm(diff2) + \
            torch.norm(diff3) + torch.norm(diff4)
        loss = loss + var_scale * loss_var

        # R_feature loss
        loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
        loss = loss + bn_reg_scale * loss_distr  # best for noise before BN

        if debug_output and epoch % 200 == 0:
            print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
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

    if idx == 0:
        print('Teacher correct out of {}: {}, loss at {}'.format(
            bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))

    name_use = "best_images"
    if prefix is not None:
        name_use = prefix + name_use
    next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1

    vutils.save_image(best_inputs[:20].clone(),
                      './{}/output_{}.png'.format(name_use, next_batch),
                      normalize=True, scale_each=True, nrow=10)

    return best_inputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=2000, type=int,
                        help='number of iterations for model inversion')
    parser.add_argument('--cig_scale', default=0.0,
                        type=float, help='competition score')
    parser.add_argument('--di_lr', default=0.1, type=float,
                        help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=0.001,
                        type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10,
                        type=float, help='weight for BN regularization statistic')
    parser.add_argument('--exp_descr', default="try1",
                        type=str, help='name to be added to experiment name')

    args = parser.parse_args([])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("loading resnet34")

    net = ResNet34()
    model_path = os.path.join(PWD, 'models/CIFAR10/net_resnet34.pt')
    save_path, load_path = utility.save_load_path(model_path, True)
    checkpoint = torch.load(load_path, map_location=device)

    net.to(device)

    criterion = nn.CrossEntropyLoss()

    # place holder for inputs
    data_type = torch.float
    inputs = torch.randn((args.bs, 3, 32, 32),
                         requires_grad=True, device=device, dtype=data_type)

    optimizer_di = optim.Adam([inputs], lr=args.di_lr)

    net.load_state_dict(checkpoint)

    batch_idx = 0
    prefix = "runs/data_generation/" + args.exp_descr + "/"

    for create_folder in [prefix, prefix + "/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    global_iteration = 0

    print("Starting model inversion")

    inputs = get_images(net=net, bs=args.bs, epochs=args.iters_mi, idx=batch_idx,
                        prefix=prefix,
                        global_iteration=global_iteration,
                        optimizer=optimizer_di, inputs=inputs, bn_reg_scale=args.r_feature_weight,
                        var_scale=args.di_var_scale, random_labels=False,
                        device=device)
