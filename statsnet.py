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
        super(StatsHook, self).__init__()

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

        print(x.shape)

        if not self.initialized:
            self.init_parameters(num_features=x.shape[1])

        if not self.enabled:
            return
        labels = self.stats_net[0].current_labels
        if self.tracking_stats:
            utility.cat_cond_mean_(x.detach(), labels, self.num_classes, self.num_features,
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

    def init_parameters(self, num_features):
        self.num_features = num_features

        if not self.class_conditional:
            num_classes = 1
        else:
            num_classes = self.num_classes
        shape = (num_classes, self.num_features)

        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.register_buffer('class_count', torch.zeros((num_classes, 1),
                                                        dtype=torch.long))
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True, bessel_correction=True):
        super(CStatsNet, self).__init__()

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
        self.net(next(iter(data_loader))[0])

    def predict(self, inputs):
        return self.net.predict(inputs)

    def stop_tracking_stats(self):
        self.eval()
        for m in self.hooks:
            m.tracking_stats = False

    def start_tracking_stats(self):
        self.enable_hooks()
        self.eval()
        for m in self.hooks:
            m.tracking_stats = True

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
