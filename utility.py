import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec

import torch

import itertools
from functools import reduce, wraps
from itertools import product
from collections.abc import Iterable
import time

from tqdm.auto import tqdm
# if sys.argv[0] == 'ipykernel_launcher':
#     from tqdm import tqdm_notebook as tqdm


class timer():
    def __init__(self):
        self.time = time.perf_counter()

    def minutes_passed(self, mins=1):
        t = time.perf_counter()
        m = (t - self.time) // 60
        if m >= mins:
            self.time = t
            return True
        return False


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.perf_counter()
        result = f(*args, **kwargs)
        run_time = time.perf_counter() - start
        print("{} took {}".format(f.__name__ + "()", prettify_time(run_time)))
        return result
    return wrap


def prettify_time(seconds):
    hours = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    sec = int((seconds % 60))
    if hours > 0:
        return "{}h {}m {}s".format(hours, mins, sec)
    elif mins > 0:
        return "{}m {}s".format(mins, sec)
    elif sec > 0:
        return "{:.2f}s".format(seconds)
    else:
        return "{}ms".format(int(seconds * 1000))


def is_iterable(x):
    return isinstance(x, Iterable)


def dict_product(params):
    return [dict(zip(params.keys(), v)) for v in product(*params.values())]


def dict_to_str(p):
    return " ".join([k + "=" + str(v) for k, v in p.items()])


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum()


def sum_all_but(x, dim):
    dims = list(range(len(x.shape)))
    dims.remove(dim)
    return x.sum(dim=dims)


def expand_as_r(a, b):
    diff = len(b.shape) - len(a.shape)
    shape = list(a.shape) + diff * [1]
    return a.reshape(shape)


def print_accuracy(X, Y, net, str_format=None):
    predictions = net.predict(X)
    accuracy = (predictions == Y).type(torch.float).mean()
    accuracy = f"{accuracy * 100:.0f}%"
    if str_format is None:
        print(f"Accuracy: {accuracy}")
    else:
        print(str_format.format(accuracy=accuracy))


def combine_mean_var(mean_a, var_a, n_a, mean_b, var_b, n_b):
    n_a = expand_as_r(n_a, mean_a)
    n_b = expand_as_r(n_b, mean_a)
    mean_a = nan_to_zero(mean_a)
    mean_b = nan_to_zero(mean_b)
    var_a = nan_to_zero(var_a)
    var_b = nan_to_zero(var_b)
    n = n_a + n_b
    mean = (n_a * mean_a + n_b * mean_b) / n
    # assert mean[n.squeeze() != 0].isfinite().all(), \
    #     "mean not finite \n{}".format(mean)
    var = (n_a * var_a
           + n_b * var_b
           + n_a * n_b / n * (mean_a - mean_b)**2) / n

    return mean, var, n


# def combine_mean_var_bn(mean_a, var_a, n_a, mean_b, var_b, n_b, momentum=0.1):
#     n = n_a + n_b
#     mean = (1 - momentum) * mean_a + (momentum) * mean_b
#     # assert mean[n.squeeze() != 0].isfinite().all(), \
#     #     "mean not finite \n{}".format(mean)
#     # use biased var in train
#     var = input.var([0, 2, 3], unbiased=False)
#     n = input.numel() / input.size(1)
#     with torch.no_grad():
#         self.running_mean = exponential_average_factor * mean\
#             + (1 - exponential_average_factor) * self.running_mean
#         # update running_var with unbiased var
#         self.running_var = exponential_average_factor * var * n / (n - 1)\
#             + (1 - exponential_average_factor) * self.running_var
#     var = (n_a * var_a
#            + n_b * var_b
#            + n_a * n_b / n * (mean_a - mean_b)**2) / n
#     return mean, var, n


def reduce_mean_var(means, vars, n):
    mean, var, n = reduce(lambda x, y: combine_mean_var(
        *x, *y), zip(means, vars, n))
    return mean, var, n


def nan_to_one_(x):
    x[x != x] = 1


def nan_to_zero_(x):
    x[x != x] = 0


def nan_to_zero(x):
    nans = x != x
    if nans.any():
        x = x.clone()
        x[nans] = 0
    return x


def batch_feature_mean_var(x, keep_dims=[1], unbiased=True):
    dims_collapse = list(range(len(x.shape)))
    for dim in keep_dims:
        dims_collapse.remove(dim)
    if dims_collapse == []:
        return x, torch.zeros_like(x, device=x.device)
    mean = x.mean(dim=dims_collapse)
    var = x.var(dim=dims_collapse, unbiased=unbiased)
    return mean, var


def c_mean_var(data, labels, num_classes=None):
    if num_classes is None:
        num_classes = labels.max().item() + 1
    shape = (num_classes, data.shape[1])
    mean = torch.zeros(shape)
    var = torch.ones(shape)
    n = torch.zeros(shape[0])

    for c in torch.unique(labels):
        c_mask = labels == c
        mean[c], var[c] = batch_feature_mean_var(data[c_mask], unbiased=False)
        n[c] = c_mask.sum()

    return mean, var, n


def class_count_(n, labels):
    for c in torch.unique(labels):
        n[c] += (labels == c).to(n.dtype).sum()


def c_mean_var_old(data, labels, shape):
    # used in iteration over batches, skip channels
    # d.shape: [n_chan, ..]
    dims_collapse = list(range(len(data.shape)))[1:-1]
    # calculate size of collapsed dims
    weight = torch.prod(torch.Tensor(list(data.shape[2:])))
    S, S_2 = torch.zeros(shape), torch.zeros(shape)
    n = torch.zeros(shape[0])
    for d, c in zip(data, labels):
        S[c] += d.sum(dims_collapse)
        S_2[c] += (d**2).sum(dims_collapse)
        n[c] += 1
    n = expand_as_r(n, S)
    mean = S / n / weight
    # print("weight, ", weight)
    # print("c_mean : ", mean)
    var = (S_2 - S**2 / n / weight) / n / weight
    # nan_to_zero(mean)
    # nan_to_zero(var)
    return mean, var, n


def assert_mean_var(calculated_mean, calculated_var, recorded_mean, recorded_var, cc_n=None):
    # check for infinites/nans where they shouldn't be
    if cc_n is not None:
        assert recorded_mean[cc_n.squeeze() != 0].isfinite().all(), \
            "recorded mean has invalid entries"
        assert recorded_var[cc_n.squeeze() != 0].isfinite().all(), \
            "recorded var has invalid entries"

    assert recorded_mean.shape == calculated_mean.shape, \
        "\ncalculated mean shape: {}".format(calculated_mean.shape) \
        + "\nrecorded mean shape: {}".format(recorded_mean.shape)
    assert recorded_var.shape == calculated_var.shape, \
        "\ncalculated var shape: {}".format(calculated_var.shape) \
        + "\nrecorded var shape: {}".format(recorded_var.shape)

    assert torch.allclose(recorded_mean, calculated_mean, atol=1e-7, equal_nan=True), (
        "\nmean max error: {}".format(
            (recorded_mean - calculated_mean).abs().max())
        + "\ncalculated mean: \n{}".format(calculated_mean)
        + "\nrecorded mean: \n{}".format(recorded_mean)
    )
    assert torch.allclose(recorded_var, calculated_var, equal_nan=True), (
        "\nvar max error: {}".format(
            (recorded_var - calculated_var).abs().max())
        + "\ncalculated var: \n{}".format(calculated_var)
        + "\nrecorded var: \n{}".format(recorded_var)
    )


def search_drive(path):
    pwd = 'Thesis'

    drive_root = path.split(pwd)[0] + 'drive/My Drive/' + pwd
    drive_path = path.replace(pwd, 'drive/My Drive/' + pwd)

    save_path, load_path = path, path

    if os.path.exists(drive_root):  # drive connected
        save_path = drive_path
        if os.path.exists(drive_path):
            load_path = drive_path

    for path in [save_path, load_path]:  # make sure directories exist
        _dir = os.path.dirname(path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    return save_path, load_path


def train(net, data_loader, criterion, optimizer,
          num_epochs=1, save_every=10,
          model_path=None, use_drive=False,
          resume_training=False):
    "Training Loop"

    if model_path is not None:
        if use_drive:
            save_path, load_path = search_drive(model_path)
        else:
            save_path, load_path = model_path, model_path
    else:
        save_path, load_path = None, None

    if load_path is not None and os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        if 'net_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['net_state_dict'])
            init_epoch = checkpoint['epoch'] + 1
        else:
            net.load_state_dict(checkpoint)
            init_epoch = 1
        print("Training Checkpoint restored: " + load_path)
        if not resume_training:
            return
    else:
        print("No Checkpoint found.")
        init_epoch = 1

    # validation = True
    # if validation:
    #     len_train = int(len(data_loader) * 0.8)
    #     len_val = len(data_loader) - len_train
    #     train_set, val_set = torch.utils.data.random_split(data_loader, [len_train, len_val])
    #     print("Splitting Dataset.")
    #     print(f"size_train {len(train_set)}, size_val {len(val_set)}")
    net.train()

    print("Beginning training.", flush=True)

    with tqdm(range(init_epoch, init_epoch + num_epochs), desc="Epoch") as t_bar:
        saved_epoch = 0
        for epoch in t_bar:
            total_count = 0.0
            total_loss = 0.0
            total_correct = 0.0

            for data in data_loader:
                inputs, labels = data

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = len(inputs)
                total_count += batch_size
                total_loss += loss.item() * batch_size
                total_correct += count_correct(outputs, labels)

                # if verbose == 1:
                #     print("{batch %d} avg-loss: %.3f, avg-accuracy: %.3f" %
                #         (total_count / batch_size, total_loss / total_count, total_correct / total_count))

            accuracy = (total_correct / total_count).item()
            # tb.add_scalar('Loss/train', total_loss, epoch)
            # tb.add_scalar('Accuracy/train', accuracy, epoch)

            # if epoch % print_every == 0:
            #     print("[%d / %d] loss: %.3f, accuracy: %.3f" %
            #         (epoch, init_epoch + num_epochs - 1, total_loss, accuracy))

            if save_path is not None \
                and (save_every is not None
                     and epoch % save_every == 0
                     or epoch == init_epoch + num_epochs - 1):
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                }, save_path)
                saved_epoch = epoch

            t_bar.set_postfix(
                loss=total_loss, accuracy=f"{accuracy*100:.0f}%", saved_epoch=saved_epoch)

    print("Finished Training")


def learn_stats(stats_net, data_loader):

    stats_net.start_tracking_stats()

    print("Beginning tracking stats.")

    with torch.no_grad(), tqdm(data_loader, desc="Batch") as t_bar:
        for data in t_bar:
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            stats_net(x)

    print("Finished Tracking Stats.")


def scatter_matrix(data, labels,
                   feature_names=[], fig=None, **kwargs):
    ndata, ncols = data.shape
    data = data.T
    if fig is None:
        fig, axes = plt.subplots(nrows=ncols, ncols=ncols, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
    else:
        axes = np.array(fig.axes).reshape(4, 4)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j), (j, i)]:
            if 'c' in kwargs:
                axes[x, y].scatter(data[x], data[y], **kwargs)
            else:
                axes[x, y].scatter(data[x], data[y],
                                   c=labels, **kwargs)
            # plot_prediction2d(data_2d, labels, net,
            #                   num=100, axis=axes[x, y], cmap=cmap)

    for i, label in enumerate(feature_names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    for i, j in zip(range(ncols), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig


def plot_contourf_data(data, func, n_grid=400, scale_grid=1, cmap='Spectral', alpha=.3, levels=None, contour=False, colorbar=False):
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    m_x = (x_min + x_max) / 2
    m_y = (y_min + y_max) / 2
    x_min = scale_grid * (x_min - m_x) + m_x
    x_max = scale_grid * (x_max - m_x) + m_x
    y_min = scale_grid * (y_min - m_y) + m_y
    y_max = scale_grid * (y_max - m_y) + m_y
    plot_contourf(x_min, x_max, y_min, y_max, func,
                  n_grid, cmap, alpha, levels, contour, colorbar)


def plot_contourf(x_min, x_max, y_min, y_max, func, n_grid=400, cmap='Spectral', alpha=.3, levels=None, contour=False, colorbar=False):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                         np.linspace(y_min, y_max, n_grid))
    mesh = (np.c_[xx.ravel(), yy.ravel()])
    mesh = torch.from_numpy(mesh.astype('float32'))
    Z = func(mesh)
    Z = Z.T.reshape(xx.shape)
    cf = plt.contourf(xx, yy, Z, levels=levels, cmap=cmap,
                      alpha=alpha, linewidths=5, linestyles='solid')
    if contour:
        if levels is not None:
            plt.contour(xx, yy, Z, cmap=cmap, levels=levels, linewidths=0.3)
        else:
            plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    if levels is not None:
        plt.colorbar(cf)


def make_grid(X, labels=None, description=None, title_fmt="label: {}", ncols=3, colors=None):
    L = len(X)
    nrows = -(-L // ncols)
    plt.figure(figsize=(ncols, nrows))

    if title_fmt is not None and title_fmt != "" and labels is not None:
        hspace = 1.0
    else:
        hspace = 0.0

    gs = gridspec.GridSpec(nrows, ncols,
                           wspace=0.0,
                           hspace=hspace,
                           #    top=1. - 0.5 / (nrows + 1), bottom=0.5 / (nrows + 1),
                           #    left=0.5 / (ncols + 1), right=1 - 0.5 / (ncols + 1)
                           )
    frame_plot = []
    for n in range(nrows):
        for m in range(ncols):
            i = n * ncols + m
            if i >= L:
                break
            ax = plt.subplot(gs[n, m])
            # plt.tight_layout()
            im = ax.imshow(X[i].squeeze(), interpolation='none')
            if labels is not None:
                color = 'k' if colors is None else colors[i]
                plt.title(title_fmt.format(labels[i]), color=color)
            plt.xticks([])
            plt.yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            frame_plot.append(im)
            if description is not None and n == nrows - 1 and m == ncols // 2:
                title = ax.text(0.22, -.3, description,
                                size=plt.rcParams["axes.titlesize"],
                                # ha="center",
                                transform=ax.transAxes
                                )
                frame_plot.append(title)
    # plt.subplots_adjust(wspace=0, hspace=0)

    return frame_plot


def make_grid_ani(frames, **kwargs):
    ani_frames = []
    for X in frames:
        frame_plot = make_grid(X, **kwargs)
        ani_frames.append(frame_plot)
    return ani_frames


def in_product_outer(nrows, ncols, i):
    x = i % nrows + 1
    y = i // nrows + 1
    return x == 1 or x == ncols or y == 1 or y == nrows


def in_product_edge(nrows, ncols, i):
    x = i % nrows + 1
    y = i // nrows + 1
    return (x == 1 and y == 1
            or x == nrows and y == 1
            or x == 1 and y == ncols
            or x == nrows and y == ncols)


def subplots_labels(x_labels, y_labels):
    for i, ax in enumerate(plt.gcf().axes):
        ax.set(xlabel=x_labels[i % len(x_labels)],
               ylabel=y_labels[i // len(y_labels)])
    for ax in plt.gcf().axes:
        ax.label_outer()


def categorical_colors(num_classes):
    cmap = matplotlib.cm.get_cmap('Spectral')
    if num_classes == 1:
        return ['k']
    return cmap(np.arange(num_classes) / (num_classes - 1.))


# cmaps = ['Reds_r', 'Blues_r']
def categorical_cmaps(num_classes):
    return [matplotlib.colors.LinearSegmentedColormap.from_list("", [categorical_colors(num_classes)[c], "white"])
            for c in range(num_classes)]


def DKL(P, Q, n_samples=1e4):
    X = P.sample(n_samples)
    return P.logpdf(X).mean() - Q.logpdf(X).mean()


def JS(P, Q, n_samples=1e4):
    M = P + Q
    return 1 / 2 * (DKL(P, M, n_samples) + DKL(Q, M, n_samples))


# stupid matplotlib....
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def plot_stats(X, Y=None):
    if Y is None:
        if isinstance(X, list):
            mean, var = [x.mean(dim=0) for x in X], [x.var(dim=0) for x in X]
        else:
            mean, var = [X.mean(dim=0)], [X.var(dim=0)]
    else:
        mean, var, _ = c_mean_var(X, Y, Y.max() + 1)
    plot_stats_mean_var(mean, var)


def plot_stats_mean_var(mean, var):
    num_classes = len(mean)
    colors = categorical_colors(num_classes)
    Ellipse = matplotlib.patches.Ellipse
    for m, v, c in zip(mean, var, colors):
        s = torch.sqrt(v) * 2
        ell = Ellipse(m, s[0], s[1], edgecolor=c, lw=1, fill=False)
        plt.gca().add_artist(ell)
        plt.scatter(m[0], m[1], color=c, edgecolors='k', marker='^')


def plot_random_projections(RP, X, Y=None, marker='o'):
    if Y is None:
        _plot_random_projections(RP, X)
    else:
        cmaps = categorical_colors(Y.max().item() + 1)
        for c in torch.unique(Y):
            _plot_random_projections(RP, X[Y == c], color=cmaps[c], marker=marker)


def _plot_random_projections(RP, X, Y=None, color='r', marker='o'):
    X_proj = X.matmul(RP)
    for rp, x_p in zip(RP.T, X_proj.T):
        m, s = x_p.mean(), x_p.var().sqrt()
        rp_m = rp * m
        start = rp_m - rp * s
        end = rp_m + rp * s
        plt.plot((start[0], end[0]), (start[1], end[1]), color=color)
        plt.plot(*rp_m, color='black', marker='x')
        _plot = plt.scatter(*(rp.reshape(-1, 1) * x_p), color=color, alpha=0.1, marker=marker)
    _plot.set_label('random projected')
    for rp_x, rp_y in RP.T:
        plt.plot((0, rp_x * 3), (0, rp_y * 3), c='black')
