import os
import sys
from collections import defaultdict

import torch
from torch.cuda.amp import autocast, GradScaler
# from scipy.stats import betabinom

from utility import tqdmEpoch

import matplotlib.pyplot as plt

# import shared
import utility

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(utility)

from debug import debug


# def betabinom_distr(N, a=1, b=1):
#     rv = betabinom(N - 1, a, b)
#     return [rv.pmf(k) for k in range(N)]


# def inversion_weights(N, slope=0, power=1, factor=1):
#     linear = [(1 + i * slope / N) for i in range(N)]
#     exponential = [l ** power for l in linear]
#     s = sum(exponential)
#     out = [factor * e / s for e in exponential]
#     return out
# @timing
def invert(data_loader, loss_fn, optimizer,
           steps=10,
           scheduler=None,
           use_amp=False,
           grad_norm_fn=None,
           callback_fn=None,
           plot=False,
           track_per_batch=False,
           track_grad_norm=False,
           ):

    assert utility.valid_data_loader(
        data_loader), f"invalid data_loader: {data_loader}"

    params = sum((p_group['params']
                  for p_group in optimizer.param_groups), [])
    device = params[0].device
    USE_AMP = (device.type == 'cuda') and use_amp
    if USE_AMP:
        scaler = GradScaler()

    metrics = defaultdict(list)
    num_batches = len(data_loader)

    def process_result(res):
        if isinstance(res, dict):
            info = res
            loss = res['loss']
        else:
            loss = res
        info = {'loss': loss.item(), **info}
        return loss, info

    print(flush=True)

    with tqdmEpoch(steps, len(data_loader)) as pbar:
        for epoch in range(steps):
            for batch_i, data in enumerate(data_loader):

                step = epoch + batch_i / num_batches

                optimizer.zero_grad()

                if USE_AMP:
                    with autocast():
                        res = loss_fn(data)
                    loss, info = process_result(res)
                    scaler.scale(loss).backward()
                    grad_scale = scaler.get_scale()
                else:
                    res = loss_fn(data)
                    loss, info = process_result(res)
                    loss.backward()
                    grad_scale = 1

                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step(loss)

                if track_grad_norm or grad_norm_fn:
                    total_norm = torch.norm(torch.stack(
                        [p.grad.detach().norm() / grad_scale for p in params])).item()

                    if grad_norm_fn:
                        rescale_coef = grad_norm_fn(total_norm) / total_norm
                        for param in params:
                            param.grad.detach().mul_(rescale_coef)

                    info['|grad|'] = total_norm

                pbar.set_postfix(**info, refresh=False)
                pbar.update()

                for k, v in info.items():
                    if batch_i == 0 or track_per_batch:
                        metrics[k].append(v)
                    else:
                        metrics[k][-1] += v

                if batch_i == 0 or track_per_batch:
                    metrics['step'].append(step)
                # batch end

            if not track_per_batch:
                for k, v in metrics.items():
                    if '[mean]' in v:
                        metrics[k][-1] /= num_batches

            if callback_fn:
                m = {k: v[-1] for k, v in metrics.items()}
                callback_fn(epoch, m)
            # epoch end

    print(flush=True)

    if plot and steps > 1:
        utility.plot_metrics(metrics, smoothing=0)
        plt.show()

    return metrics


def inversion_loss(stats_net, criterion, target_labels, hp,
                   layer_weights=None, regularization=None,
                   reg_reduction_type='mean'):
    # if layer_weights is None:
    #     layer_weights = betabinom_distr(
    #         len(stats_net.hooks) - 1, hp['distr_a'], hp['distr_b'])

    def loss_fn(x):
        stats_net.set_reg_reduction_type(reg_reduction_type)
        if len(target_labels) == 1:
            targets = target_labels.repeat(len(x))
        else:
            targets = target_labels
        outputs = stats_net({'inputs': x, 'labels': targets})
        criterion_loss = criterion(outputs, targets)

        r_input, r_components = stats_net.get_hook_regularizations()
        if layer_weights is not None:
            r_layer = sum([w * c for w, c in zip(layer_weights, r_components)])
        else:
            r_layer = sum(r_components)

        info = {}
        loss = torch.Tensor([0])
        if hp['factor_input'] != 0.0:
            loss = loss + hp['factor_input'] * r_input
        if hp['factor_layer'] != 0.0:
            loss = loss + hp['factor_layer'] * r_layer
        if hp['factor_reg'] != 0.0 and regularization is not None:
            loss = loss + hp['factor_reg'] * regularization(x)
        if hp['factor_criterion'] != 0.0:
            loss = loss + hp['factor_criterion'] * criterion_loss

        if reg_reduction_type != "none":
            info['accuracy'] = (torch.argmax(outputs, dim=1)
                                == targets).to(torch.float).mean().item()
            if hp['factor_input'] != 0.0:
                info['r_input'] = r_input.item()
            if hp['factor_layer'] != 0.0:
                info['r_layer'] = r_layer.item()
            if hp['factor_reg'] != 0.0 and regularization is not None:
                info['reg'] = regularization.item()
            if hp['factor_criterion'] != 0.0:
                info['criterion'] = criterion_loss.item()
            return loss, info

        return loss
    return loss_fn
