import os
import sys
import torch
import torch.nn as nn

import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utility
importlib.reload(utility)


class StatsHook(nn.Module):

    def __init__(self, stats_net, module, num_classes,
                 class_conditional=True, bessel_correction=False):
        super().__init__()

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = [stats_net]
        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.bessel_correction = bessel_correction
        self.tracking_stats = True
        self.initialized = False
        self.enabled = False

    def hook_fn(self, module, inputs, outputs):

        x = inputs[0]

        if not self.initialized:
            self.init_parameters(x.shape[1:])

        if not self.enabled:
            return
        labels = self.stats_net[0].current_labels
        if self.tracking_stats:
            utility.cat_cond_mean_(x.detach(), labels,
                                   mean=self.running_mean,
                                   var=self.running_var,
                                   cc=self.class_count,
                                   class_conditional=self.class_conditional,
                                   bessel_correction=self.bessel_correction)
        else:   # inverting
            if not self.initialized:
                print("Error: Statistics Parameters not initialized")
            means = self.running_mean[labels]
            self.regularization = (x - means).norm(2).sum()

    def init_parameters(self, shape):
        if self.class_conditional:
            num_classes = self.num_classes
        else:
            num_classes = 1

        self.shape = [num_classes] + list(shape)

        self.register_buffer('running_mean', torch.zeros(self.shape))
        self.register_buffer('running_var', torch.zeros(self.shape))
        self.register_buffer('class_count',
                             utility.expand_as_r(
                                 torch.zeros(num_classes, dtype=torch.long),
                                 self.running_mean))
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True, bessel_correction=True):
        super().__init__()

        self.net = net

        self.hooks = nn.ModuleList()
        for i, (name, m) in enumerate(net.named_modules()):
            if isinstance(m, nn.ModuleList) or isinstance(m, nn.CrossEntropyLoss):
                continue
            if i == 0:  # XXX always assume this is neural net??
                continue
            self.hooks.append(StatsHook(self, m, num_classes,
                                        class_conditional=class_conditional,
                                        bessel_correction=bessel_correction))

        # self.class_conditional = class_conditional

    def forward(self, data):
        if isinstance(data, dict):
            self.current_labels = data['labels']
            return self.net(data['inputs'])
        else:
            return self.net(data)

    def init_hooks(self, data_loader):
        inputs = next(iter(data_loader))[0]
        self.net(inputs)
        self.input_shape = inputs.shape[1:]

    def predict(self, inputs):
        return self.net.predict(inputs)

    def stop_tracking_stats(self):
        self.eval()
        for h in self.hooks:
            h.tracking_stats = False

    def start_tracking_stats(self):
        self.eval()
        self.enable_hooks()
        for h in self.hooks:
            h.tracking_stats = True

    def enable_hooks(self):
        for h in self.hooks:
            h.enabled = True

    def disable_hooks(self):
        for h in self.hooks:
            h.enabled = False

    def collect_stats(self):
        stat_vars = ['running_mean', 'running_var', 'class_count']
        stats = []
        for m in self.hooks:
            stat = {}
            for s in stat_vars:
                stat[s] = getattr(m, s).data
            stats.append(stat)
        return stats
