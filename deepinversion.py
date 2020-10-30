import utility
from utility import timing
import shared

from torch.cuda.amp import autocast, GradScaler
from scipy.stats import betabinom

try:
    from apex import amp
    USE_AMP = True
except ImportError:
    USE_AMP = False


def betabinom_distr(N, a=1, b=1):
    rv = betabinom(N - 1, a, b)
    return [rv.pmf(k) for k in range(N)]


# def inversion_weights(N, slope=0, power=1, factor=1):
#     linear = [(1 + i * slope / N) for i in range(N)]
#     exponential = [l ** power for l in linear]
#     s = sum(exponential)
#     out = [factor * e / s for e in exponential]
#     return out

def inversion_loss(stats_net, criterion, target_labels,
                   layer_weights=None, regularization=None,
                   reg_reduction_type='sum',
                   **hp):
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

        components = stats_net.get_hook_regularizations()
        input_reg = components.pop(0)
        # layer_reg = sum([w * c for w, c in zip(layer_weights, components)])
        layer_reg = sum(components)

        criterion_loss = hp['factor_criterion'] * criterion_loss

        total_loss = 0
        if hp['factor_input'] != 0.0:
            total_loss += hp['factor_input'] * input_reg
        if hp['factor_layer'] != 0.0:
            total_loss += hp['factor_layer'] * layer_reg
        if hp['factor_reg'] != 0.0 and regularization is not None:
            total_loss += hp['factor_reg'] * regularization(x)

        total_loss += criterion_loss

        return total_loss
    return loss_fn


@timing
def deep_inversion(inputs,
                   stats_net,
                   loss_fn,
                   optimizer,
                   steps=5,
                   perturbation=None,
                   projection=None,
                   track_history=False,
                   track_history_every=1,
                   print_every_n_min=1,
                   ):

    writer = shared.get_summary_writer()
    timer = utility.timer()

    stats_net.stop_tracking_stats()

    scaler = GradScaler()

    if track_history:
        history = [(inputs.detach().cpu().clone(), 0)]

    inputs_orig = inputs
    inputs.requires_grad = True

    for step in range(1, steps + 1):

        inputs = inputs_orig

        if perturbation is not None:
            inputs = perturbation(inputs)

        optimizer.zero_grad()
        stats_net.zero_grad()  # ??? why

        with autocast():
            loss = loss_fn(inputs)

        # if USE_AMP:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        scaler.scale(loss).backward()

        # scaler.unscale_(optimizer)
        writer.add_scalar('grad_norm',
                          (inputs_orig.grad / scaler.get_scale()).norm(2), step)

        scaler.step(optimizer)
        scaler.update()

        if projection is not None:
            inputs = projection(inputs)

        if track_history and (step % track_history_every == 0 or step == steps):
            history.append((inputs.detach().cpu().clone(), step))

        if timer.minutes_passed(print_every_n_min):
            print(f"It {step}\t Losses: total: {loss.item():3.3f}")

    if track_history:
        return history
    return [(inputs.detach().cpu().clone(), step)]
