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

    def __init__(self, stats_net, module, name=None):
        super().__init__()

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = [stats_net]
        self.initialized = False

        if name is not None:
            self.name = name

    def state(self):    # hack, so that stats_net won't be saved in state_dict recursively
        return self.stats_net[0]

    def hook_fn(self, module, inputs, outputs):

        x = inputs[0]

        if not self.initialized:
            self.init_parameters(x.shape[1:])
            return

        if not self.state().enabled:
            return

        labels = self.state().current_labels
        if self.state().tracking_stats:
            # pylint: disable=access-member-before-definition

            shape = self.running_mean.shape
            new_mean, new_var, m = utility.c_mean_var(
                x.detach(), labels, shape)

            old_mean, old_var, n = self.running_mean, self.running_var, self.class_count

            self.running_mean, self.running_var, self.class_count = utility.combine_mean_var(
                old_mean, old_var, n, new_mean, new_var, m)
        else:   # inverting
            assert self.initialized, "Statistics Parameters not initialized"

            if self.state().bn_masked and not isinstance(module, nn.BatchNorm2d):
                self.regularization = torch.Tensor([0]).to(x.device)
                return

            if self.state().method == 'paper':

                nch = x.shape[1]

                mean = x.mean([0, 2, 3])
                var = x.permute(1, 0, 2, 3).contiguous().view(
                    [nch, -1]).var(1, unbiased=False)

                r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
                    module.running_mean.data.type(var.type()) - mean, 2)

                self.regularization = r_feature
            else:
                m, v = self.running_mean[labels], self.running_var[labels]

                if not self.state().class_conditional:
                    if not hasattr(self, 'total_mean'):
                        self.total_mean, self.total_var = utility.reduce_mean_var(
                            m, v, self.class_count)
                    m, v = self.total_mean.unsqueeze(
                        0), self.total_var.unsqueeze(0)
                    self.regularization = (
                        (x.mean([2, 3]) - m)**2 / v).sum(dim=1)
                else:
                    self.regularization = (
                        (x.mean([2, 3]) - m[labels])**2 / v[labels]).sum(dim=1)
                # print("x shape: ", x.shape)
                # [64, 3, 32, 32]
                # print("m shape: ", m.shape)
                # [10, 3]
                # x.mean([2, 3])
                # [64, 3]
                # m[labels]
                # [64, 3]
                # [64]

                # print("reg is fin: ", torch.isfinite(self.regularization).all())
                if self.state().reg_reduction == 'mean':    # should use mean over batches
                    self.regularization = self.regularization.mean()
                if self.state().reg_reduction == 'sum':
                    self.regularization = self.regularization.sum()
                if self.state().reg_reduction == 'none':
                    pass

    def init_parameters(self, shape):

        num_classes = self.state().num_classes

        # CCC
        n_chan = shape[0]
        shape = [num_classes, n_chan]

        self.register_buffer('running_mean', torch.zeros(
            shape, requires_grad=False))
        self.register_buffer('running_var', torch.zeros(
            shape, requires_grad=False))
        self.register_buffer('class_count',
                             utility.expand_as_r(
                                 torch.zeros(num_classes, requires_grad=False),
                                 self.running_mean))
        self.initialized = True


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True):
        super().__init__()

        self.net = net

        self.hooks = nn.ModuleDict()

        for i, (name, m) in enumerate(net.named_modules()):
            if (i != 0 and
                (isinstance(m, nn.ModuleList)
                 or isinstance(m, nn.ModuleDict)
                 or isinstance(m, nn.Sequential)
                 or isinstance(m, nn.CrossEntropyLoss)
                 or isinstance(m, nn.modules.activation.ReLU)
                 )):
                continue
            if i == 0:
                name = "net_input"
                continue    # input tracked double
            assert len(m._forward_hooks.values()
                       ) == 0, "module already has hook"
            if name == "":
                name = str(i)
            hook_name = name.replace('.', '-')
            self.hooks[hook_name] = StatsHook(self, m, name=hook_name)
            # print("{} adding hook to module {} of type {}".format(i, name, type(m)))

        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.reg_reduction = 'mean'
        self.tracking_stats = True
        self.enabled = True
        self.bn_masked = False
        self.method = 'standard'

    def forward(self, data):
        if isinstance(data, dict):
            self.current_labels = data['labels']
            return self.net(data['inputs'])
        else:
            return self.net(data)

    def init_hooks(self, init_sample):
        self.net(init_sample)
        self.input_shape = init_sample.shape[1:]

    def get_hook_regularizations(self):
        components = [h.regularization for h in self.hooks.values()]
        return components

    def predict(self, inputs):
        self.disable_hooks()
        with torch.no_grad():
            outputs = self.net.predict(inputs)
        self.enable_hooks()
        return outputs

    def stop_tracking_stats(self):
        self.eval()
        self.tracking_stats = False

    def start_tracking_stats(self):
        self.eval()
        self.tracking_stats = True

    def set_reg_reduction_type(self, reg_type):
        self.reg_reduction = reg_type

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
        self.enabled = True

    def disable_hooks(self):
        self.enabled = False
