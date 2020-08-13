import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import shared
import deepinversion

import importlib
importlib.reload(utility)
importlib.reload(deepinversion)

# tb = shared.get_summary_writer()


class StatsHook(nn.Module):

    def __init__(self, stats_net, module, num_classes,
                 class_conditional=True):
        super().__init__()

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = [stats_net]
        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.reg_reduction = "mean"
        self.tracking_stats = True
        self.initialized = False
        self.enabled = True

    def hook_fn(self, module, inputs, outputs):

        x = inputs[0]

        if not self.initialized:
            self.init_parameters(x.shape[1:])
            return

        if not self.enabled:
            return

        labels = self.stats_net[0].current_labels
        if self.tracking_stats:
            # pylint: disable=access-member-before-definition
            new_mean, new_var, m = utility.c_mean_var(
                x.detach(), labels, self.shape)
            old_mean, old_var, n = self.running_mean, self.running_var, self.class_count
            self.running_mean, self.running_var, self.class_count = utility.combine_mean_var(
                old_mean, old_var, n,
                new_mean, new_var, m)
        else:   # inverting
            assert self.initialized, "Statistics Parameters not initialized"

            means = self.running_mean[labels]
            vars = self.running_var[labels]
            if not self.class_conditional:
                if not hasattr(self, 'total_mean'):
                    self.total_mean, self.total_var = utility.reduce_mean_var(
                        means, vars, self.class_count)
                means, vars = self.total_mean, self.total_var
            self.regularization = utility.sum_all_but(
                (x - means)**2 / vars, dim=0)
            if self.reg_reduction == "mean":    # should use mean over batches
                self.regularization = self.regularization.mean()
            if self.reg_reduction == "sum":
                self.regularization = self.regularization.sum()
            # self.regularization = ((x - means) / vars.sqrt()).norm(2).sum()

    def init_parameters(self, shape):

        num_classes = self.num_classes

        self.shape = [num_classes] + list(shape)

        self.register_buffer('running_mean', torch.zeros(
            self.shape, requires_grad=False))
        self.register_buffer('running_var', torch.zeros(
            self.shape, requires_grad=False))
        self.register_buffer('class_count',
                             utility.expand_as_r(
                                 torch.zeros(num_classes, requires_grad=False),
                                 self.running_mean))
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, hook_before_bn=False, track_inputs=True, class_conditional=True):
        super().__init__()

        self.net = net

        self.hooks = nn.ModuleDict()

        for i, (name, m) in enumerate(net.named_modules()):
            if i == 0 and not track_inputs:
                continue
            if (i != 0 and
                (isinstance(m, nn.ModuleList)
                 or isinstance(m, nn.ModuleDict)
                 or isinstance(m, nn.Sequential)
                 or isinstance(m, nn.CrossEntropyLoss))):
                continue
            if i != 0 and hook_before_bn and not isinstance(m, nn.BatchNorm2d):
                continue
            if i == 0:
                name = "net_input"
            # print("{} adding hook to module ".format(i) + name)
            hook_name = name.replace('.', '-')
            if name == "":
                name = str(i)
            self.hooks[hook_name] = StatsHook(self, m, num_classes,
                                              class_conditional=class_conditional)

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
        self.disable_hooks()
        with torch.no_grad():
            outputs = self.net.predict(inputs)
        self.enable_hooks()
        return outputs

    def inversion_loss(self, inputs, target, weights, criterion, reduction='mean'):
        if utility.is_iterable(target):
            labels = target
        else:
            labels = torch.LongTensor([target] * len(inputs))
        data = {'inputs': inputs, 'labels': labels}
        self.set_reg_reduction_type(reduction)
        loss = deepinversion.inversion_loss(data, self, weights, criterion)
        self.set_reg_reduction_type('mean')
        return loss

    def stop_tracking_stats(self):
        self.eval()
        for h in self.hooks.values():
            h.tracking_stats = False

    def start_tracking_stats(self):
        '''tracking_stats state is similar to self.training in eval()/train()
           it is decoupled though, so statistics can be collected while BN stats are frozen
        '''
        self.eval()
        for h in self.hooks.values():
            h.tracking_stats = True

    def set_reg_reduction_type(self, type):
        for h in self.hooks.values():
            h.reg_reduction = type

    def collect_stats(self):
        stat_vars = ['running_mean', 'running_var', 'class_count']
        stats = []
        for h in self.hooks.values():
            stat = {}
            for s in stat_vars:
                stat[s] = getattr(h, s).data
            stats.append(stat)
        return stats

    def enable_hooks(self):
        for h in self.hooks.values():
            h.enabled = True

    def disable_hooks(self):
        for h in self.hooks.values():
            h.enabled = False
