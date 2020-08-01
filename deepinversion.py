import torch
import torch.optim as optim


def deep_inversion(stats_net, criterion, labels, steps=5, track_history=False):

    stats_net.stop_tracking_stats()
    stats_net.enable_hooks()

    shape = [len(labels)] + list(stats_net.input_shape)
    inputs = torch.randn(shape, requires_grad=True)

    optimizer = optim.Adam([inputs], lr=0.1)

    if track_history:
        history = []
        history.append(inputs.data.detach().clone())

    for step in range(1, steps + 1):

        optimizer.zero_grad()

        data = {'inputs': inputs, 'labels': labels}
        outputs = stats_net(data)
        loss = criterion(outputs, labels)

        # reg_loss = sum([s.regularization for s in net.stats])
        # reg_loss = stats_net.hooks[0].regularization

        # loss = loss + reg_loss
        loss = stats_net.hooks[0].regularization

        loss.backward()

        optimizer.step()

        if track_history:
            history.append(inputs.data.detach().clone())

    print("Finished Inverting")

    stats_net.disable_hooks()

    if track_history:
        return history
    return inputs.data.detach()
