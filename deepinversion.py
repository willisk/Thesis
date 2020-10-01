from utility import timing

from scipy.stats import betabinom


def betabinom_distr(N, a=1, b=1):
    rv = betabinom(N - 1, a, b)
    return [rv.pmf(k) for k in range(N)]


# def inversion_weights(N, slope=0, power=1, factor=1):
#     linear = [(1 + i * slope / N) for i in range(N)]
#     exponential = [l ** power for l in linear]
#     s = sum(exponential)
#     out = [factor * e / s for e in exponential]
#     return out


@timing
def deep_inversion(inputs,
                   stats_net,
                   loss_fn,
                   optimizer,
                   steps=5,
                   perturbation=None,
                   projection=None,
                   track_history=False, track_history_every=1):

    # tb = shared.get_summary_writer()

    stats_net.stop_tracking_stats()

    if projection is not None:
        inputs = projection(inputs)

    inputs.requires_grad = True

    # tb.add_scalar('DI/slope', slope)
    # tb.add_scalar('DI/', accuracy, epoch)

    if track_history:
        history = []
        history.append((inputs.detach().clone(), 0))

    for step in range(1, steps + 1):

        if perturbation is not None:
            inputs = perturbation(inputs)

        optimizer.zero_grad()
        stats_net.net.zero_grad()

        loss = loss_fn(inputs)

        loss.backward()

        optimizer.step()

        if projection is not None:
            inputs = projection(inputs)

        if track_history and (step % track_history_every == 0 or step == steps):
            history.append((inputs.detach().clone(), step))

    print("Finished Inverting")

    if track_history:
        return history
    return [(inputs.detach(), step)]
