from matplotlib.axes._axes import _log as matplotlib_axes_logger
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import gridspec

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import inspect
from functools import reduce, wraps

import itertools
from itertools import product
from collections import defaultdict
from collections.abc import Iterable
import time

from tqdm import tqdm

from debug import debug


def tqdm_fmt_dict(epochs, batch_size):
    total = epochs * batch_size
    return dict(
        total=total,
        bar_format=f"{{l_bar}}{{bar}}|{{n:.1f}}/{str(epochs)} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]",
        unit_scale=1 / batch_size,
        unit='epoch',
    )


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


def dict_to_str(p, delim=" "):
    return delim.join([f"{k}={v}" for k, v in p.items()])


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum().item()


def sum_all_but(x, dim):
    dims = list(range(len(x.shape)))
    dims.remove(dim)
    return x.sum(dim=dims)


def expand_as_r(a, b):
    diff = len(b.shape) - len(a.shape)
    shape = list(a.shape) + diff * [1]
    return a.reshape(shape)


def net_accuracy(net, data_loader, inputs_pre_fn=None):
    total_count = 0.0
    total_correct = 0.0
    device = next(iter(net.parameters())).device
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs_pre_fn:
                inputs = inputs_pre_fn(inputs)
            outputs = net(inputs)
            total_count += len(inputs)
            total_correct += count_correct(outputs, labels)
    return total_correct / total_count


def print_net_accuracy(net, data_loader):
    accuracy = net_accuracy(net, data_loader)
    print(f"net accuracy: {accuracy * 100:.1f}%")


def net_accuracy_batch(net, inputs, labels):
    with torch.no_grad():
        outputs = net(inputs)
        return count_correct(outputs, labels) / len(inputs)


def print_net_accuracy_batch(net, inputs, labels):
    accuracy = net_accuracy_batch(net, inputs, labels)
    print(f"net accuracy: {accuracy * 100:.1f}%")


def exp_av_mean_var(m_a, v_a, m_b, v_b, gamma):
    mean = gamma * m_a + (1 - gamma) * m_b
    var = (gamma * v_a
           + (1 - gamma) * v_b
           + gamma * (1 - gamma) * (m_a - m_b)**2)
    return mean, var


def combine_mean_var(m_a, v_a, n_a, m_b, v_b, n_b, cap_gamma=1):
    # if class_conditional:
    #     n_a = expand_as_r(n_a, m_a)
    #     n_b = expand_as_r(n_b, m_a)
    #     m_a = nan_to(m_a, 0)
    #     m_b = nan_to(m_b, 0)
    #     v_a = nan_to(v_a, 0)
    #     v_b = nan_to(v_b, 0)
    n = n_a + n_b
    gamma = expand_as_r(torch.clamp(n_a / n, max=cap_gamma), m_a)
    mean, var = exp_av_mean_var(m_a, v_a, m_b, v_b, gamma)
    return mean, var, n


def reduce_stats(stats, n):
    if isinstance(stats, list):
        return [reduce(lambda x, y: combine_mean_var(*x, *y),
                    zip(mean, var, n))[:2] for mean, var in stats]
    return reduce(lambda x, y: combine_mean_var(*x, *y),
                  zip(mean, var, n))[:2]


def nan_to_one_(x):
    x[x != x] = 1


def nan_to_zero_(x):
    x[x != x] = 0


def nan_to(x, num):
    nans = x != x
    if nans.any():
        x = x.clone()
        x[nans] = num
    return x


# def class_count_(n, labels):
#     for c in torch.unique(labels):
#         n[c] += (labels == c).to(n.dtype).sum()


# @debug
def batch_feature_stats(X, std=False, keepdim=False):
    dims_collapse = list(range(X.ndim))
    dims_collapse.remove(1)
    assert dims_collapse != [], "dims to collapse are empty"
    mean = X.mean(dim=dims_collapse, keepdim=keepdim)
    valid_mean = len(X) > 0
    valid_var = len(X) > 1
    if not valid_mean:
        mean = torch.zeros_like(mean)
    if not valid_var:
        var = torch.zeros_like(mean)
    else:
        if std:
            var = X.std(dim=dims_collapse, unbiased=False, keepdim=keepdim)
        else:
            var = X.var(dim=dims_collapse, unbiased=False, keepdim=keepdim)
    return mean, var


# @debug
def c_stats(inputs, labels, n_classes, return_count=False, std=False, keepdim=False):
    mean = var = None
    n = torch.zeros(n_classes, dtype=torch.long, device=inputs.device)

    for c in labels.unique().long():
        c_mask = labels == c
        mean_c, var_c = batch_feature_stats(
            inputs[c_mask], std=std, keepdim=keepdim)
        if mean is None:
            shape = (n_classes,) + mean_c.shape
            mean = torch.zeros(shape, dtype=inputs.dtype, device=inputs.device)
            var = torch.ones(shape, dtype=inputs.dtype, device=inputs.device)
        mean[c], var[c] = mean_c, var_c

        n[c] = c_mask.sum().item()

    if return_count:
        return mean, var, n
    return mean, var


# @debug
def get_stats(inputs, labels=None, n_classes=None, class_conditional=False, std=False, return_count=False, keepdim=False, dtype=torch.float):
    if isinstance(inputs, list):
        return [_get_stats(x.to(dtype), labels, n_classes, class_conditional, std, return_count, keepdim) for x in inputs]
    return _get_stats(inputs.to(dtype), labels, n_classes, class_conditional, std, return_count, keepdim)


# @debug
def _get_stats(inputs, labels=None, n_classes=None, class_conditional=False, std=False, return_count=False, keepdim=False):
    if class_conditional:
        assert labels is not None and n_classes is not None
        return c_stats(inputs, labels, n_classes,
                       std=std, return_count=return_count, keepdim=keepdim)
    if return_count:
        m = torch.LongTensor([len(inputs)]).to(inputs.device)
        return (*batch_feature_stats(inputs, std=std, keepdim=keepdim), m)
    return batch_feature_stats(inputs, std=std, keepdim=keepdim)


def collect_min_max(data_loader, device='cpu', path=None, use_drive=True):

    def min_max(data):
        inputs, labels = data
        return inputs.min().item(), inputs.max().item()

    def accumulate_fn(old, new):
        return min(old[0], new[0]), max(old[1], new[1])

    return collect_data(data_loader, min_max, accumulate_fn,
                        map_location=device, path=path, use_drive=use_drive)


# @debug
def collect_stats(data_loader, projection, n_classes, class_conditional,
                  std=False, keepdim=False,
                  device='cpu', path=None, use_drive=True):

    def data_fn(data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = projection((inputs, labels))
        stats = get_stats(outputs, labels, n_classes, class_conditional,
                          std=False, return_count=True, keepdim=keepdim, dtype=torch.double)
        return stats

    def update_fn(old, new):
        if isinstance(old, list):
            return [combine_mean_var(*o, *n) for o, n in zip(old, new)]
        return combine_mean_var(*old, *new)

    stats = collect_data(data_loader, data_fn, update_fn,
                         map_location=device, path=path, use_drive=use_drive)

    if isinstance(stats, list):
        return [(m.float(), v.sqrt().float() if std else v.float()) for m, v, _ in stats]
    return stats[0].float(), stats[1].sqrt().float() if std else stats[1].float()

# @debug


def collect_data(data_loader, data_fn, accumulate_fn,
                 final_fn=None, map_location='cpu', path=None, use_drive=True):

    save_path, load_path = save_load_path(path, use_drive=use_drive)

    print(load_path)
    if load_path and os.path.exists(load_path):
        print(f"Loading data from {load_path}.", flush=True)
        return torch.load(load_path, map_location=map_location)

    print(flush=True)

    output = None
    with torch.no_grad(), tqdm(data_loader, unit="batch") as pbar:
        for data in pbar:
            if output is None:
                output = data_fn(data)
            else:
                output = accumulate_fn(output, data_fn(data))

    if final_fn:
        output = final_fn(output)

    print(flush=True)
    if save_path:
        torch.save(output, save_path)
        print(f"Saving data to {save_path}.")
    return output


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


def save_load_path(path, use_drive=True):
    if path is not None:
        if use_drive:
            save_path, load_path = search_drive(path)
        else:
            save_path, load_path = path, path
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_path, load_path
    return None, None


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


def valid_data_loader(data_loader):
    return isinstance(data_loader, torch.utils.data.DataLoader) or isinstance(data_loader, list)


def train(net, data_loader, criterion, optimizer,
          epochs=10, save_every=20,
          model_path=None, use_drive=False,
          resume_training=False, reset=False,
          scheduler=None, plot=False, use_amp=False):
    "Training Loop"

    device = next(net.parameters()).device

    save_path, load_path = save_load_path(model_path, use_drive)

    if load_path is not None and not reset and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        if 'net_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['net_state_dict'], strict=False)
            init_epoch = checkpoint['epoch'] + 1
        else:
            net.load_state_dict(checkpoint)
            init_epoch = 1
        print("Training Checkpoint restored: " + load_path)
        if not resume_training:
            net.eval()
            return
    else:
        print("No Checkpoint found / Reset.")
        if save_path:
            print("Path: " + save_path)
        init_epoch = 1

    assert valid_data_loader(
        data_loader), f"invalid data_loader: {data_loader}"

    net.train()

    USE_AMP = device.type == 'cuda' and use_amp
    if USE_AMP:
        scaler = GradScaler()

    TRACKING = None
    if plot:
        TRACKING = defaultdict(list, loss=[])

    print("Beginning training.", flush=True)

    with tqdm(**tqdm_fmt_dict(epochs, len(data_loader))) as pbar:
        saved_epoch = 0
        for epoch in range(init_epoch, init_epoch + epochs):
            total_count = 0.0
            total_loss = 0.0
            total_correct = 0.0
            grad_total = 0.0

            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                if USE_AMP:
                    with autocast():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    grad_scale = scaler.get_scale()
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    grad_scale = 1

                for param in net.parameters():
                    grad_total += (param.grad.norm(2) / grad_scale).item()

                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                batch_size = len(inputs)
                total_count += batch_size
                total_loss += loss.item() * batch_size
                total_correct += count_correct(outputs, labels)
                pbar.set_postfix(
                    loss=total_loss / total_count,
                    acc=f"{total_correct / total_count * 100:.0f}%",
                    chkpt=saved_epoch,
                    refresh=False,
                )
                pbar.update()

            loss = total_loss / total_count
            accuracy = total_correct / total_count
            grad_norm = grad_total  # / total_count

            if scheduler is not None:
                scheduler.step(grad_norm)

            if TRACKING:
                TRACKING['loss'].append(loss)
                TRACKING['accuracy'].append(accuracy)
                TRACKING['grad'].append(grad_norm)

            if save_path is not None \
                and (save_every is not None
                     and epoch % save_every == 0
                     or epoch == init_epoch + epochs - 1):
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                }, save_path)
                saved_epoch = epoch

            pbar.set_postfix(
                loss=total_loss / total_count,
                acc=f"{total_correct / total_count * 100:.0f}%",
                chkpt=saved_epoch,
            )

    print(flush=True, end='')
    net.eval()

    if TRACKING:
        plot_metrics(TRACKING, step_start=init_epoch)
        plt.xlabel('epochs')
        plt.show()
        return TRACKING


def sgm(x, sh=1):
    return np.exp(np.log(x + sh).sum() / len(x)) - sh


def plot_metrics(metrics, step_start=1):
    if 'step' in metrics:
        steps = metrics.pop('step')
    else:
        steps = range(step_start, len(metrics['loss']) + 1)

    accuracy = None
    if 'accuracy' in metrics:
        accuracy = metrics.pop('accuracy')

    for key, val in metrics.items():
        plt.plot(steps, val, label=key)

    vals = np.nan_to_num(np.array(sum(metrics.values(), [])), nan=10000)
    # print("min", vals.min())
    # print("max", vals.max())
    # sgm_m = sgm(vals, sh=vals.max())
    # sgm_s = sgm(np.abs(vals - sgm_m), sh=vals.max())
    sgm_m = sgm(vals)
    sgm_s = sgm(np.abs(vals - sgm_m))
    y_min, y_max = min(vals), max(vals)
    y_min, y_max = max(y_min, sgm_m - sgm_s), min(y_max, sgm_m + sgm_s)
    buffer = 0.1 * (y_max - y_min)
    plt.gca().set_ylim([y_min - buffer, y_max + buffer])

    if accuracy:
        acc_scaled = [y_min + a * (y_max - y_min) for a in accuracy]
        plt.plot(steps, acc_scaled, alpha=0.6, label='acc(scaled)')

    plt.title('metrics')
    plt.xlabel('steps')
    plt.legend()


def learn_stats(stats_net, data_loader):

    stats_net.start_tracking_stats()

    print("Beginning tracking stats.", flush=True)

    with torch.no_grad(), tqdm(data_loader, desc="Batch") as pbar:
        for data in pbar:
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            stats_net(x)

    print(flush=True)


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
                      alpha=alpha, linestyles='solid')
    if contour:
        if levels is not None:
            plt.contour(xx, yy, Z, cmap=cmap, levels=levels, linewidths=0.3)
        else:
            plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    if levels is not None:
        plt.colorbar(cf)

    # MC Integration
    # d_x = torch.as_tensor(x_max - x_min) / n_grid
    # d_y = torch.as_tensor(y_max - y_min) / n_grid
    # I = ((torch.as_tensor(Z)).sum() * d_x * d_y).item()
    # print(f"MC Integral: {I}")


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
    return cmap(np.arange(num_classes) / (num_classes - 1.))


# cmaps = ['Reds_r', 'Blues_r']
def categorical_cmaps(num_classes):
    return [matplotlib.colors.LinearSegmentedColormap.from_list("", [categorical_colors(num_classes)[c], "white"])
            for c in range(num_classes)]


# def DKL(P, Q, n_samples=1e4):
#     X = P.sample(n_samples)
#     return P.logpdf(X).mean() - Q.logpdf(X).mean()


# def JS(P, Q, n_samples=1e4):
#     M = P + Q
#     return 1 / 2 * (DKL(P, M, n_samples) + DKL(Q, M, n_samples))

def logsumexp(a, dim=None, b=None):
    a = torch.as_tensor(a)
    a_max = a.max().item() if dim is None else torch.max(a, dim=dim)[0]
    if b is not None:
        e = torch.as_tensor(b) * torch.exp(a - a_max)
        out = e.sum().log() if dim is None else e.sum(dim=dim).log()
    else:
        out = (a - a_max).exp().sum(dim=0).log()
    return out + a_max


# stupid matplotlib....
matplotlib_axes_logger.setLevel('ERROR')


def plot_stats(X, Y=None, colors=None):
    if Y is None:
        if isinstance(X, list):
            mean, var = [x.mean(dim=0) for x in X], [x.var(dim=0) for x in X]
        else:
            mean, var = [X.mean(dim=0)], [X.var(dim=0)]
    else:
        mean, var, _ = c_mean_var(X, Y, Y.max() + 1)
    plot_stats_mean_var(mean, var, colors=colors)


def plot_stats_mean_var(mean, var, colors=None):
    if colors is None:
        colors = categorical_colors(len(mean))
    Ellipse = matplotlib.patches.Ellipse
    for m, v, c in zip(mean, var, colors):
        s = torch.sqrt(v) * 2
        ell = Ellipse(m, s[0], s[1], edgecolor=c, lw=1, fill=False)
        plt.gca().add_artist(ell)
        plt.scatter(m[0], m[1], color=c, edgecolors='k', marker='^')


def plot_random_projections(RP, X_proj, mean, Y=None, color='r', marker='o', scatter=True):
    if Y is None:
        _plot_random_projections(
            RP, X_proj, mean=mean, color=color, marker=marker, scatter=scatter,)
    else:
        n_classes = len(mean)
        if n_classes == 2:
            marker = ['+', 'd']
        # cmaps = categorical_colors(Y.max().item() + 1)
        for c, m in zip(Y.unique(), marker):
            _plot_random_projections(RP, X_proj[Y == c],
                                     mean=mean[c],
                                     color=color,
                                     marker=m,
                                     scatter=scatter,)


def _plot_random_projections(RP, X_proj, mean, color='r', marker='o', scatter=True):
    for rp, x_p in zip(RP.T, X_proj.T):
        m, s = x_p.mean(), x_p.var().sqrt()
        rp_m = rp * m + mean
        start, end = rp_m - rp * s, rp_m + rp * s
        plt.plot(*list(zip(start, end)), color=color)
        plt.plot(*rp_m, color='black', marker='x')
        if scatter:
            X_proj_abs = mean + rp * x_p.reshape(-1, 1)
            _plot = plt.scatter(*X_proj_abs.T,
                                color=color, alpha=0.1, marker=marker)
    if scatter:
        _plot.set_label('random projected')
    # for rp in RP.T:
    #     plt.plot(*list(zip(mean, mean + rp * 3)), c='black')


def print_tabular(data, row_name="", spacing=2):
    print()
    headers = list(dict.fromkeys([k for d in data.values() for k in d.keys()]))
    row_data = ([[row_name] + headers] +
                [[m] + [f"{data[m][h]:.2f}" if h in data[m] else "N.A."
                        for h in headers]
                 for m in data.keys()])
    widths = [max(map(len, column)) for column in zip(*row_data)]
    for i, rd in enumerate(row_data):
        line = "".join(f"{e:<{w + spacing}}" for e, w in zip(rd, widths))
        print(line)
        if i == 0:
            print('-' * len(line))


def get_child_modules(net):
    ignore_types = ['activation', 'loss', 'container', 'batchnorm', 'pooling']
    all_layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0:
            skip = False
            for ignore in ignore_types:
                if ignore in layer.__module__:
                    skip = True
            if skip:
                continue
            all_layers.append(layer)
        else:
            all_layers += get_child_modules(layer)
    return all_layers


def get_bn_layers(net):
    ignore_types = ['activation', 'loss', 'container', 'pooling']
    all_layers = []
    for layer in net.children():
        if len(list(layer.children())) == 0:
            skip = False
            for ignore in ignore_types:
                if ignore in layer.__module__:
                    skip = True
            if skip:
                continue
            if 'batchnorm' in layer.__module__:
                all_layers.append(layer)
        else:
            all_layers += get_bn_layers(layer)
    return all_layers


def get_num_params(layers):
    total = 0
    for l in layers:
        for p in l.parameters():
            if p.requires_grad:
                total += p.numel()
    return total
