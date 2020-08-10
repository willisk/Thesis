import torch
import torch.optim as optim

import shared


def reg_weighting(components, slope=1, power=1):
    N = len(components)  # div by N doesn't matter if normalizing
    linear = [c * (i * slope / N) for i, c in enumerate(components, 1)]
    exponential = [l ** power for l in linear]
    # total = sum(components)
    # total = sum(linear)
    total = sum(exponential)
    total = total / total.detach()
    return total


def deep_inversion(stats_net, criterion, labels,
                   steps=5, lr=0.1,
                   track_history=False, track_history_every=1):

    tb = shared.get_summary_writer()

    stats_net.stop_tracking_stats()
    stats_net.enable_hooks()

    shape = [len(labels)] + list(stats_net.input_shape)
    inputs = torch.randn(shape, requires_grad=True)

    optimizer = optim.Adam([inputs], lr=lr)

    # tb.add_scalar('DI/slope', slope)
    # tb.add_scalar('DI/', accuracy, epoch)

    if track_history:
        history = []
        history.append(inputs.detach().clone())

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = stats_net(data)
        loss = criterion(outputs, labels)

        components = [h.regularization for h in stats_net.hooks.values()]
        reg_loss = reg_weighting(components)
        # reg_loss = stats_net.hooks[-1].regularization
        # loss = stats_net.hooks[-1].regularization

        loss = loss + reg_loss

        loss.backward()

        optimizer.step()

        if track_history and (step % track_history_every == 0 or step == steps):
            history.append(inputs.detach().clone())

    print("Finished Inverting")

    stats_net.disable_hooks()

    if track_history:
        return history
    return inputs.detach()
