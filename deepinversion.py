import shared

import torch
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import betabinom

from tqdm.auto import tqdm


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
def deep_inversion(inputs,
                   loss_fn,
                   optimizer,
                   steps=5,
                   pre_fn=None,
                   track_history=False,
                   track_history_every=1,
                   ):

    writer = shared.get_summary_writer()
    # timer = utility.timer()
    info = {}

    USE_AMP = inputs.device.type == 'gpu'
    if USE_AMP:
        scaler = GradScaler()

    if track_history:
        history = [(inputs.detach().cpu().clone(), 0)]
    else:
        history = []

    inputs_orig = inputs

    print("Beginning Inversion.", flush=True)

    with tqdm(range(1, steps + 1), desc="Step") as t_bar:
        for step in t_bar:

            inputs = inputs_orig

            if pre_fn is not None:
                inputs = pre_fn(inputs)

            optimizer.zero_grad()

            if USE_AMP:
                with autocast():
                    result = loss_fn(inputs)
                    if isinstance(result, tuple):
                        loss, info = result
                    else:
                        loss = result
                scaler.scale(loss).backward()
                grad_scale = scaler.get_scale()
            else:
                result = loss_fn(inputs)
                if isinstance(result, tuple):
                    loss, info = result
                else:
                    loss = result
                loss.backward()
                grad_scale = 1

            # writer.add_scalar('loss', loss.item(), step)
            # writer.add_scalar(
            #     'gradient_norm', (inputs_orig.grad / grad_scale).norm(2), step)

            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if track_history and (step % track_history_every == 0 or step == steps):
                history.append((inputs.detach().cpu().clone(), step))

            # if timer.minutes_passed(print_every_n_min):
            #     print(f"Step {step}\t loss: {loss.item():3.3f}")
            t_bar.set_postfix(loss=loss.item(), **info)

    print(flush=True)

    return history
