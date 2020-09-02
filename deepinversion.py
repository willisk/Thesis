import numpy as np

import torch
import torch.optim as optim

import shared

from scipy.stats import betabinom


def inversion_layer_weights(N, a=1, b=1):
    rv = betabinom(N - 1, a, b)
    return [rv.pmf(k) for k in range(N)]


def inversion_loss_weights(weights, factor):
    if factor == 0:
        return weights + [0]
    elif factor == np.inf:
        return [0 for w in weights] + [1]
    else:

        return [w / (1 + factor) for w in weights] + [factor / (1 + factor)]


# def inversion_weights(N, slope=0, power=1, factor=1):
#     linear = [(1 + i * slope / N) for i in range(N)]
#     exponential = [l ** power for l in linear]
#     s = sum(exponential)
#     out = [factor * e / s for e in exponential]
#     return out

def inversion_loss(data, stats_net, weights, criterion):
    outputs = stats_net(data)
    loss = criterion(outputs, data['labels'])
    components = [h.regularization for h in stats_net.hooks.values()]
    reg_loss = sum([w * c for w, c in zip(weights[:-1], components)])
    return weights[-1] * loss + reg_loss


def deep_inversion(stats_net, criterion, labels,
                   steps=5, lr=0.1,
                   weights=None,
                   perturbation=None,
                   regularization=None,
                   projection=None,
                   track_history=False, track_history_every=1):

    # tb = shared.get_summary_writer()

    stats_net.stop_tracking_stats()

    shape = [len(labels)] + list(stats_net.input_shape)
    inputs = torch.randn(shape, requires_grad=True)

    optimizer = optim.Adam([inputs], lr=lr)

    # tb.add_scalar('DI/slope', slope)
    # tb.add_scalar('DI/', accuracy, epoch)

    if track_history:
        history = []
        history.append(inputs.detach().clone())

    if weights is None:
        weights = inversion_layer_weights(N=len(stats_net.hooks))
        weights = inversion_loss_weights(weights, 1)

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        if perturbation is not None:
            inputs = perturbation(inputs)

        loss = stats_net.inversion_loss(
            inputs, labels, weights, criterion, reduction='mean')

        if regularization is not None:
            loss = loss + regularization(inputs)

        loss.backward()

        optimizer.step()

        if projection is not None:
            with torch.no_grad():
                inputs = projection(inputs)

        if track_history and (step % track_history_every == 0 or step == steps):
            history.append(inputs.detach().clone())

    print("Finished Inverting")

    if track_history:
        return history
    return inputs.detach()
