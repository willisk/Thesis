import shared
import utility
import importlib
importlib.reload(utility)

from collections import defaultdict

import torch
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import betabinom

from tqdm.auto import tqdm

import matplotlib.pyplot as plt


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
def deep_inversion(data_loader, loss_fn, optimizer, steps=10,
                   pre_fn=None, scheduler=None,
                   #    track_history_every=None,
                   plot=False,
                   device='cpu',
                   ):

    # writer = shared.get_summary_writer()
    USE_AMP = (device == 'gpu')
    if USE_AMP:
        scaler = GradScaler()

    # history = []

    TRACKING = False
    if plot:
        TRACKING = defaultdict(list)

    def process_result(res, metrics):
        info = {}
        if isinstance(res, tuple):
            loss, info = res
            for k, v in info.items():
                metrics[k] += v
        else:
            loss = res
        metrics['loss'] += loss.item()
        return loss

    print("Beginning Inversion.", flush=True)

    with tqdm(range(1, steps + 1), desc="Epoch") as pbar:

        for step in pbar:

            METRICS = defaultdict(float)

            for inputs, labels in data_loader:
                labels = labels.to(inputs.device)
                # inputs = inputs_orig
                # if step == 1 and track_history_every:
                #     history = [(inputs.detach().cpu().clone(), 0)]

                if pre_fn is not None:
                    inputs = pre_fn(inputs)

                optimizer.zero_grad()

                if USE_AMP:
                    with autocast():
                        res = loss_fn(inputs, labels)
                        loss = process_result(res, METRICS)
                    scaler.scale(loss).backward()
                    grad_scale = scaler.get_scale()
                else:
                    res = loss_fn(inputs, labels)
                    loss = process_result(res, METRICS)
                    loss.backward()
                    grad_scale = 1

                grad_total = 0.
                if TRACKING:
                    for p_group in optimizer.param_groups:
                        for i, param in enumerate(p_group['params']):
                            # p_name = f"parpam_{'-'.join(map(str, param.shape))}"
                            p_name = f'grad_{i}'
                            p_grad = (param.grad / grad_scale).norm(2).item()
                            grad_total += p_grad
                            METRICS[p_name] += p_grad

                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step(grad_total)

            n_batches = len(data_loader)
            for k in METRICS:
                METRICS[k] /= n_batches

            pbar.set_postfix(**METRICS)

            if TRACKING:
                for k, v in METRICS.items():
                    TRACKING[k].append(v)

            # if track_history_every and (
            #         step % track_history_every == 0 or step == steps):
            #     history.append((inputs.detach().cpu().clone(), step))

    print(flush=True, end='')

    if TRACKING:
        utility.plot_metrics(TRACKING)
        plt.show()

    # return history
