import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility
import deepinversion

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(utility)
    importlib.reload(deepinversion)


from functools import wraps


def no_hooks(func):
    @wraps(func)
    def _no_hooks(self, *args, **kwargs):
        enabled = self.enabled
        self.enabled = False
        out = func(self, *args, **kwargs)
        self.enabled = enabled
        return out
    return _no_hooks


class StatsHook(nn.Module):

    def __init__(self, stats_net, module, name=None, post_hook=False):
        super().__init__()

        self.hook = module.register_forward_hook(self.hook_fn)

        self.stats_net = [stats_net]
        self.modulex = [module]
        self.initialized = False
        self.post_hook = post_hook

        if name is not None:
            self.name = name

    def state(self):    # hack, so that stats_net won't be saved in state_dict recursively
        return self.stats_net[0]

    def hook_fn(self, module, inputs, outputs):

        x = outputs if self.post_hook else inputs[0]

        STATE = self.state()

        if not self.initialized:
            self.init_parameters(x.shape[1])
            # print(self.name, "input shape: ", x.shape)

        if not STATE.enabled:
            return

        # if isinstance(module, nn.BatchNorm2d):
        #     m, v = self.running_mean, self.running_var
        #     m, v = utility.reduce_mean_var(
        #         m, v, self.class_count)

        #     print("allclose mean: ", torch.allclose(m, module.running_mean))
        #     print("diff : ", torch.norm(m - module.running_mean))
        #     # print("allclose var: ", torch.allclose(v, module.running_var))

        # pylint: disable=access-member-before-definition
        labels = STATE.current_labels
        n_class = self.running_mean.shape[0]

        if STATE.tracking_stats:

            new_mean, new_var, m = utility.c_mean_var(
                x.detach(), labels, n_class)

            old_mean, old_var, n = self.running_mean, self.running_var, STATE.class_count

            # if False: #using momentum
            #     momentum = 0.1

            #     self.running_mean = momentum * \
            #         new_mean + (1 - momentum) * old_mean

            #     n_el = x.numel() / x.shape[0] / x.shape[1]
            #     self.running_var = momentum * new_var * n_el / (n_el - 1) \
            #         + (1 - momentum) * old_var

            new_mean, new_var, n = utility.combine_mean_var(
                old_mean, old_var, n, new_mean, new_var, m)

            self.running_mean, self.running_var, _ = new_mean, new_var, n

        else:   # inverting
            assert self.initialized, "Statistics Parameters not initialized"

            ########### INVERTING #############

            if STATE.bn_masked and not isinstance(module, nn.BatchNorm2d):
                self.regularization = torch.Tensor([0]).to(x.device)
                return

            if STATE.method == 'paper':
                if not STATE.class_conditional:
                    # batch_mean = x.mean([0, 2, 3])
                    batch_mean, batch_var = utility.batch_feature_mean_var(x,
                                                                           keep_dims=[1])
                    running_mean, running_var = self.reduced_mean_var()

                    # (n_feat) = (n_feat) - (n_feat)
                    diff_mean = running_mean - batch_mean
                    diff_var = running_var - batch_var
                    r_feature = (torch.norm(diff_mean, 2)
                                 + torch.norm(diff_var, 2))

                else:
                    # (n_class, n_feat)
                    cc_mean, cc_var, _ = utility.c_mean_var(
                        x, labels, n_class)

                    # XXX: Don't take mean, should be reduce_mean_var
                    # (n_feat) = (n_class, n_feat) - (n_class, n_feat)
                    diff_mean = (cc_mean - self.running_mean).mean(dim=0)
                    diff_var = (running_var - batch_var).mean(dim=0)
                    r_feature = (torch.norm(diff_mean, 2)
                                 + torch.norm(diff_var, 2))
                    self.regularization = r_feature

                    # =========================
            else:   # NOT PAPER
                # shape: (bs, n_feat)
                bs_mean, bs_var = utility.batch_feature_mean_var(x,
                                                                 keep_dims=[0, 1])
                if not STATE.class_conditional:
                    running_mean, running_var = self.reduced_mean_var()
                    # (bs, n_feat) = (bs, n_feat) - (bs, 1)
                    bs_diff = bs_mean - self.running_mean[labels]
                    bs_var_diff = bs_var - self.running_var[labels]
                else:
                    # (bs, n_feat) = (bs, n_feat) - (bs, n_feat)
                    bs_diff = bs_mean - self.running_mean[labels]
                    bs_var_diff = bs_var - self.running_var[labels]
                    # XXX: sqrt for comparability?
                    # (bs)
                r_feature = ((bs_diff)**2 / self.running_var[labels]
                             #  + (bs_var_diff**2)
                             ).sum(dim=1)
                if STATE.reg_reduction != 'none':
                    r_feature = r_feature.mean()  # .sqrt()

                self.regularization = r_feature

            # if STATE.class_conditional:
            #     if STATE.method == 'paper':
            #         # (n_class, n_feat)
            #         cc_mean, cc_var, _ = utility.c_mean_var(
            #             x, labels, shape)

            #         # XXX: Don't take mean, should be reduce_mean_var
            #         # (n_feat) = (n_class, n_feat) - (n_class, n_feat)
            #         diff_mean = (cc_mean - self.running_mean).mean(dim=0)
            #         diff_var = (running_var - batch_var).mean(dim=0)
            #         r_feature = (torch.norm(diff_mean, 2)
            #                      + torch.norm(diff_var, 2))
            #         self.regularization = r_feature
            #     else:
            #         # shape: (bs, n_feat)
            #         bs_mean, bs_var = self.running_var[labels]

            #         # shape: (bs, n_feat)
            #         # XXX: Should partition batches by class and match statistics
            #         bs_mean, bs_var = utility.batch_feature_mean_var(x,
            #                                                          keep_dims=[0, 1])

            #         # (bs, n_feat) = (bs, n_feat) - (bs, n_feat)
            #         bs_diff = bs_mean - self.running_mean[labels]
            #         bs_var_diff = bs_var - self.running_var[labels]
            #         # XXX: sqrt for comparability?
            #         r_feature = ((bs_diff)**2 / self.running_var[labels]
            #                      + (bs_var_diff**2).mean(dim=1))
            #         if STATE.reg_reduction != 'none':
            #             r_feature = r_feature.mean(dim=0)  # .sqrt()
            # else:
            #     if STATE.use_bn_stats and isinstance(module, nn.BatchNorm2d):
            #         running_mean, running_var = module.running_mean, module.running_var
            #     else:
            #         self.reduce_mean_var()
            #         running_mean, running_var = self.reduced_mean, self.reduced_var
            #         # (n_feat)

            #     if STATE.method == 'paper':
            #         # batch_mean = x.mean([0, 2, 3])
            #         batch_mean, batch_var = utility.batch_feature_mean_var(x,
            #                                                                keep_dims=[1])
            #         # (n_feat)

            #         # (n_feat) = (n_feat) - (n_feat)
            #         diff_mean = running_mean - batch_mean
            #         diff_var = running_var - batch_var
            #     else:
            #         # (bs, n_feat)
            #         bs_mean, bs_var = utility.batch_feature_mean_var(x,
            #                                                          keep_dims=[0, 1])
            #         # (bs, n_feat) = (n_feat) - (bs, n_feat)
            #         diff_mean = running_mean - bs_mean
            #         diff_var = running_var - bs_var
            #     r_feature = (torch.norm(diff_mean, 2)
            #                  + torch.norm(diff_var, 2))

            # # # necessary?
            # # running_mean = running_mean.data.type(batch_mean.type())
            # # running_var = running_var.data.type(batch_var.type())

            # if STATE.method == 'paper':
            #     # scalar
            #     if STATE.class_conditional:
            #         diff_mean = (running_mean - batch_mean).mean(dim=0)
            #         diff_var = (running_var - batch_var).mean(dim=0)
            #     else:
            #         diff_mean = running_mean - batch_mean
            #         diff_var = running_var - batch_var
            #     r_feature = (torch.norm(diff_mean, 2)
            #                  + torch.norm(diff_var, 2))
            #     self.regularization = r_feature
            # else:
            #     # either [n_chan] or [BS, n_chan]
            #     r_feature = (batch_mean - running_mean)**2 / running_var
            #     if STATE.class_conditional:
            #         # [BS]
            #         r_feature = r_feature.sum(dim=1)

            #         # self.regularization = (
            #         #     (x.mean([2, 3]) - running_mean)**2 / running_var).sum(dim=1)

            #     self.regularization = r_feature

            #     # print("reg is fin: ", torch.isfinite(self.regularization).all())
            #     if STATE.reg_reduction == 'mean':    # should use mean over batches
            #         self.regularization = self.regularization.mean()
            #     if STATE.reg_reduction == 'sum':
            #         self.regularization = self.regularization.sum()
            #     if STATE.reg_reduction == 'none':
            #         pass

    def reduced_mean_var(self):
        STATE = self.state()
        module = self.modulex[0]
        if STATE.use_bn_stats and isinstance(module, nn.BatchNorm2d):
            return module.running_mean, module.running_var
        if not hasattr(self, 'reduced_mean'):
            self.reduced_mean, self.reduced_var, _ = utility.reduce_mean_var(
                self.running_mean, self.running_var, STATE.class_count)
        return self.reduced_mean, self.reduced_var

    def init_parameters(self, n_features):

        num_classes = self.state().num_classes

        shape = (num_classes, n_features)

        self.register_buffer('running_mean', torch.zeros(
            shape, requires_grad=False))
        self.register_buffer('running_var', torch.ones(
            shape, requires_grad=False))
        self.initialized = True

    def reset(self):
        self.running_mean.fill_(0)
        self.running_var.fill_(0)


class CStatsNet(nn.Module):

    def __init__(self, net, num_classes, class_conditional=True):
        super().__init__()

        self.net = net

        self.hooks = nn.ModuleDict()

        first = True

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
                continue    # input tracked double
            assert len(m._forward_hooks.values()
                       ) == 0, "module already has hook"
            if first:
                name = "net_input"
                first = False
            if name == "":
                name = str(i)
            hook_name = name.replace('.', '-')
            self.hooks[hook_name] = StatsHook(self, m, name=hook_name)
            # print("{} adding hook to module {} of type {}".format(i, name, type(m)))

        self.hooks["logits_output"] = StatsHook(
            self, m, name=hook_name, post_hook=True)

        self.num_classes = num_classes
        self.class_conditional = class_conditional
        self.enabled = True
        self.tracking_stats = False
        self.bn_masked = False
        self.method = 'standard'
        self.reg_reduction = 'mean'
        self.register_buffer('class_count', torch.zeros(
            num_classes, requires_grad=False, dtype=torch.long))

    # def stats_summary(self):
    #     import matplotlib.pyplot as plt
    #     for name, h in self.hooks.items():
    #         mean, var = h.running_mean, h.running_var
    #         x = range(mean.shape[1])
    #         print(f"layer {name}")
    #         plt.errorbar(x, mean.mean(axis=0), yerr=mean.var(
    #             axis=0), fmt='.b', capsize=0)
    #         plt.errorbar(x, var.mean(axis=0), yerr=var.var(
    #             axis=0), fmt='.', color='orange', capsize=0)
    #         plt.show()
    #         # print(f"feature_mean: {mean.mean()} variance: {mean.var()}")
    #         # print(f"feature_var mean: {var.mean()} variance: {var.var()}")

    def forward(self, data):
        if isinstance(data, dict):
            self.current_labels = data['labels']
            self.new_count = torch.zeros_like(self.class_count)
            utility.class_count_(self.new_count, data['labels'])
            output = self.net(data['inputs'])
            self.class_count += self.new_count
            return output
        else:
            print("Forward without hooks")
            return no_hooks(self.net(data))

    def init_hyperparameters(self, hp):
        self.bn_masked = hp['mask_bn']
        self.use_bn_stats = hp['use_bn_stats']
        self.class_conditional = hp['cc']
        self.method = hp['method']

    @no_hooks
    def init_hooks(self, init_sample):
        self.net(init_sample)
        self.input_shape = init_sample.shape[1:]

    def get_hook_regularizations(self):
        r_components = [h.regularization for h in self.hooks.values()]
        r_input = r_components.pop(0)  # should search for name
        return r_input, r_components

    @no_hooks
    def predict(self, inputs):
        return self.net.predict(inputs)

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

    def disable_hooks(self, func=None):
        self.enabled = False

    # def verify_hooks_finite(self):
    #     for m in self.parameters():
    #         assert h.running_mean.isfinite().all(), "{h.name} not finite mean"
    #         assert h.running_var.isfinite().all(), "{h.name} not finite var"

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
