import os
import sys
from collections import defaultdict

import torch
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import betabinom

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

# import shared
import utility

if 'ipykernel_launcher' in sys.argv or 'COLAB_GPU' in os.environ:
    import importlib
    importlib.reload(utility)

from utility import debug


def betabinom_distr(N, a=1, b=1):
    rv = betabinom(N - 1, a, b)
    return [rv.pmf(k) for k in range(N)]


# def inversion_weights(N, slope=0, power=1, factor=1):
#     linear = [(1 + i * slope / N) for i in range(N)]
#     exponential = [l ** power for l in linear]
#     s = sum(exponential)
#     out = [factor * e / s for e in exponential]
#     return out

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


# @timing
def deep_inversion(data_loader, loss_fn, optimizer,
                   steps=10,
                   data_pre_fn=None,
                   inputs_pre_fn=None,
                   scheduler=None,
                   #    track_history_every=None,
                   plot=False,
                   use_amp=False,
                   grad_multiplier_fn=None,
                   ):

    assert utility.valid_data_loader(
        data_loader), f"invalid data_loader: {data_loader}"

    # writer = shared.get_summary_writer()
    device = optimizer.param_groups[0]['params'][0].device
    USE_AMP = (device.type == 'cuda') and use_amp
    if USE_AMP:
        scaler = GradScaler()

    # history = []

    TRACKING = defaultdict(list)

    def process_result(res, metrics_acc, batch_size):
        if isinstance(res, tuple):
            loss, info = res
            display_info = {**info}
        else:
            loss = res
            display_info = {}
        display_info['loss'] = loss.item()
        for k, v in display_info.items():
            metrics_acc[k] += v / batch_size
        pbar.set_postfix(**display_info, refresh=False)
        pbar.update()
        return loss

    print("Beginning Inversion.", flush=True)

    # with tqdm(range(1, steps + 1), desc="Epoch") as pbar:
    with tqdm(**utility.tqdm_fmt_dict(steps, len(data_loader))) as pbar:

        for epoch in range(1, steps + 1):

            METRICS = defaultdict(float)
            total_count = 0

            for data in data_loader:

                # if step == 1 and track_history_every:
                #     history = [(inputs.detach().cpu().clone(), 0)]

                if isinstance(data, torch.Tensor):
                    batch_size = len(data)
                else:
                    batch_size = len(data[0])

                if data_pre_fn is not None:
                    data = data_pre_fn(data)

                if inputs_pre_fn is not None:
                    inputs, labels = data
                    inputs = inputs_pre_fn(inputs)
                    data = inputs, labels

                optimizer.zero_grad()

                if USE_AMP:
                    with autocast():
                        res = loss_fn(data)
                        loss = process_result(res, METRICS, batch_size)
                    scaler.scale(loss).backward()
                    grad_scale = scaler.get_scale()
                else:
                    res = loss_fn(data)
                    loss = process_result(res, METRICS, batch_size)
                    loss.backward()
                    grad_scale = 1

                grad_total = 0.
                # if grad_penalty:
                # params = []
                for p_group in optimizer.param_groups:
                    #     params += p_group['params']
                    for i, param in enumerate(p_group['params']):
                        lr = p_group['lr']
                        # p_name = f"parpam_{'-'.join(map(str, param.shape))}"
                        grad_norm = (param.grad / grad_scale).norm()
                        if grad_multiplier_fn:
                            # param.grad.data.mul_(grad_penalty * grad_norm.sqrt())
                            lambd = grad_multiplier_fn(grad_norm) / grad_norm
                            param.grad.data.mul_(lambd)
                        grad_total += grad_norm
                        METRICS[f"grad_{i}"] += grad_norm.item()
                # if params:
                #     for p in params:
                #         p.grad.data.mul_(0)
                #     torch.nn.utils.clip_grad_norm_(params, 0)
                # if grad_penalty:
                #     grad_total *= grad_penalty
                #     grad_total.backward()

                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step(grad_total)

                # total_count += bs
                # for k in METRICS:
                #     METRICS[k] *= bs / total_count

            for k, v in METRICS.items():
                TRACKING[k].append(v * len(data_loader))

            # if track_history_every and (
            #         step % track_history_every == 0 or step == steps):
            #     history.append((inputs.detach().cpu().clone(), step))

    print(flush=True, end='')

    if plot and TRACKING:
        utility.plot_metrics(TRACKING)
        plt.show()

    # return history
    return TRACKING
