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


def invert(data_loader, loss_fn, optimizer,
           steps=10,
           scheduler=None,
           use_amp=False,
           grad_norm_fn=None,
           callback_fn=None,
           plot=False,
           track_per_batch=False,
           track_grad_norm=False,
           print_grouped=False,
           ):

    assert utility.valid_data_loader(
        data_loader), f"invalid data_loader: {data_loader}"

    params = [(p_group['params']
               for p_group in optimizer.param_groups)]
    lrs = [(p_group['lr']
            for p_group in optimizer.param_groups)]
    device = params[0].device
    USE_AMP = (device.type == 'cuda') and use_amp
    if USE_AMP:
        scaler = GradScaler()

    metrics = defaultdict(list)
    num_batches = len(data_loader)

    def process_result(res):
        if isinstance(res, dict):
            loss = res['loss']
            info = res
        else:
            loss = res
            info = {}
        info = {**info, 'loss': loss.item()}
        return loss, info

    print(flush=True)

    if callback_fn:
        callback_fn(0, None)

    with tqdmEpoch(steps, len(data_loader)) as pbar:
        for epoch in range(steps):
            for batch_i, data in enumerate(data_loader):

                if track_per_batch:
                    step = epoch + (batch_i + 1) / num_batches
                else:
                    step = epoch + 1 + batch_i / num_batches

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
                    # XXX: probably shouldn't multiply with lr
                    total_norm = torch.norm(torch.stack(
                        [p.grad.detach().norm() / grad_scale * lr
                            for p, lr in zip(params, lrs)])).item()

                    if grad_norm_fn:
                        rescale_coef = grad_norm_fn(total_norm) / total_norm
                        for param in params:
                            param.grad.detach().mul_(rescale_coef)

                    info['|grad|'] = total_norm

                pbar.set_postfix(**{
                    k.split(']')[-1].split(':')[-1].strip(): v
                    for k, v in info.items() if ']' not in k
                }, refresh=False)
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
                    if ':mean:' in k:
                        metrics[k][-1] /= num_batches

            if callback_fn:
                callback_fn(epoch + 1, metrics)
            # epoch end

    print(flush=True)

    if plot and steps > 1:
        utility.plot_metrics(metrics, smoothing=0)
        plt.show()

    return metrics
