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
        self.modulex = [module]
        self.initialized = False

        if name is not None:
            self.name = name

    def state(self):    # hack, so that stats_net won't be saved in state_dict recursively
        return self.stats_net[0]

    def hook_fn(self, module, inputs, outputs):

        x = inputs[0]

        STATE = self.state()
        # print("outputs shape", x.shape)
        # if isinstance(module, nn.BatchNorm2d):
        #     print("hook {} input :\n\t".format(
        #         self.name), x[0, 0, 0, 0].item())

        if not self.initialized:
            self.init_parameters(x.shape[1])
            # print(self.name, "input shape: ", x.shape)

        if not STATE.enabled:
            return

        # x = outputs

        # if isinstance(module, nn.BatchNorm2d):
        #     m, v = self.running_mean, self.running_var
        #     m, v = utility.reduce_mean_var(
        #         m, v, self.class_count)

        #     print("allclose mean: ", torch.allclose(m, module.running_mean))
        #     print("diff : ", torch.norm(m - module.running_mean))
        #     # print("allclose var: ", torch.allclose(v, module.running_var))

        labels = STATE.current_labels

        if STATE.tracking_stats:
            # pylint: disable=access-member-before-definition

            shape = self.running_mean.shape

            new_mean, new_var, m = utility.c_mean_var_bn(
                x.detach(), labels, shape)

            old_mean, old_var, n = self.running_mean, self.running_var, self.class_count

            self.class_count = n + m

            if False:
                momentum = 0.1

                self.running_mean = momentum * \
                    new_mean + (1 - momentum) * old_mean

                n_el = x.numel() / x.shape[0] / x.shape[1]
                self.running_var = momentum * new_var * n_el / (n_el - 1) \
                    + (1 - momentum) * old_var
            else:
                mean, var, n = utility.combine_mean_var(
                    old_mean, old_var, n, new_mean, new_var, m)

                self.running_mean, self.running_var, self.class_count = mean, var, n

        else:   # inverting
            assert self.initialized, "Statistics Parameters not initialized"

            if STATE.bn_masked and not isinstance(module, nn.BatchNorm2d):
                self.regularization = torch.Tensor([0]).to(x.device)
                return

            if STATE.class_conditional:
                # shape: [BS, n_chan]
                running_mean, running_var = self.running_mean[labels], self.running_var[labels]

                # shape: [BS, n_chan]
                batch_mean = x.mean([2, 3])
                batch_var = x.var([2, 3])
            else:
                if not hasattr(self, 'reduced_mean'):
                    self.reduced_mean, self.reduced_var, _ = utility.reduce_mean_var(
                        self.running_mean, self.running_var, self.class_count)

                # shape: [n_chan]
                running_mean, running_var = self.reduced_mean, self.reduced_var

                if STATE.use_bn_stats and isinstance(module, nn.BatchNorm2d):
                    running_mean, running_var = module.running_mean, module.running_var

                # shape: [n_chan]
                batch_mean = x.mean([0, 2, 3])
                batch_var = x.var([0, 2, 3])

            # necessary?
            running_mean = running_mean.data.type(batch_mean.type())
            running_var = running_var.data.type(batch_var.type())

            if STATE.method == 'paper':
                # scalar
                r_feature = (torch.norm(running_mean - batch_mean, 2)
                             + torch.norm(running_var - batch_var, 2))
            else:
                # either [n_chan] or [BS, n_chan]
                r_feature = (batch_mean - running_mean)**2 / running_var
                if STATE.class_conditional:
                    # [BS]
                    r_feature = r_feature.sum(dim=1)

                    # self.regularization = (
                    #     (x.mean([2, 3]) - running_mean)**2 / running_var).sum(dim=1)

                self.regularization = r_feature

                # print("reg is fin: ", torch.isfinite(self.regularization).all())
                if STATE.reg_reduction == 'mean':    # should use mean over batches
                    self.regularization = self.regularization.mean()
                if STATE.reg_reduction == 'sum':
                    self.regularization = self.regularization.sum()
                if STATE.reg_reduction == 'none':
                    pass

    def init_parameters(self, n_features):

        num_classes = self.state().num_classes

        shape = (num_classes, n_features)

        self.register_buffer('running_mean', torch.zeros(
            shape, requires_grad=False))
        self.register_buffer('running_var', torch.ones(
            shape, requires_grad=False))
        self.register_buffer('class_count',
                             utility.expand_as_r(
                                 torch.zeros(num_classes, requires_grad=False),
                                 self.running_mean))
        self.initialized = True

    def reset(self):
        self.running_mean.fill_(0)
        self.running_var.fill_(0)
        self.class_count.fill_(0)


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
        self.reg_reduction = 'sum'
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
        self.enabled = False
        self.net(init_sample)
        self.enabled = True
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

    def verify_hooks_finite(self):
        for h in self.hooks.values():
            assert h.running_mean.isfinite().all(), "{h.name} not finite mean"
            assert h.running_var.isfinite().all(), "{h.name} not finite var"
            # m = h.modulex[0]
            # if isinstance(m, torch.nn.BatchNorm2d):
            #     print("layer {}".format(h.name))
            #     h_mean, h_var, h_cc = utility.reduce_mean_var(
            #         h.running_mean, h.running_var, h.class_count)
            #     print("\tmean max error: {}".format(
            #         (h_mean - m.running_mean).abs().max()))
            #     print("\tvar max error: {}".format(
            #         (h_var - m.running_var).abs().max()))
