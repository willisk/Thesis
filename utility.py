import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch

import itertools
from functools import reduce, wraps
from itertools import product
from collections.abc import Iterable
import time


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


def combine_mean_var(mean_a, var_a, n_a, mean_b, var_b, n_b):
    n = n_a + n_b
    mean = (n_a * mean_a + n_b * mean_b) / n
    var = (n_a * var_a
           + n_b * var_b
           + n_a * n_b / n * (mean_a - mean_b)**2) / n

    return mean, var, n


def reduce_mean_var(means, vars, n):
    return reduce(lambda x, y: combine_mean_var(*x, *y), zip(means, vars, n))


def nan_to_zero(x):
    x[x != x] = 0


def c_mean_var(data, labels, shape):
    S = torch.zeros(shape, requires_grad=False)
    S_2 = torch.zeros(shape, requires_grad=False)
    n = torch.zeros(shape[0], requires_grad=False)
    for d, c in zip(data, labels):
        S[c] += d
        S_2[c] += d**2
        n[c] += 1
    n = expand_as_r(n, S)
    mean = S / n
    var = (S_2 - S**2 / n) / n
    nan_to_zero(mean)
    nan_to_zero(var)
    return mean, var, n


# def cat_cond_mean_(inputs, labels, mean, var, cc,
#                    class_conditional=True):

#     shape = mean.shape
#     n_classes = shape[0]

#     if class_conditional:
#         total = torch.zeros(shape)
#         total.index_add_(0, labels, inputs)
#         N_class = expand_as_r(torch.bincount(labels, minlength=n_classes), cc)
#     else:
#         total = inputs.sum(dim=0)
#         N_class = len(inputs)

#     cc_f = cc.float()
#     cc.add_(N_class)

#     mean.mul_(cc_f / cc)
#     mean.add_((total / cc).expand_as(mean))
#     ##
#     total_2 = torch.zeros(shape)
#     # total_var.index_add_(0, labels, inputs - mean[labels])
#     for i, x in enumerate(inputs):
#         total_2[labels[i]] += (inputs[i])**2
#     var.mul_(cc_f / cc)
#     # var.add_((total_2 - N_class * mean**2) / cc)
#     var.add_((total_2 - mean**2) / cc)


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
          num_epochs=1, print_every=1, save_every=1,
          model_path=None, use_drive=False,
          resume_training=False, verbose=0):
    "Training Loop"

    net.train()

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

    print("Beginning training.")

    for epoch in range(init_epoch, init_epoch + num_epochs):

        saved_epoch = False

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

            if verbose == 1:
                print("{batch %d} avg-loss: %.3f, avg-accuracy: %.3f" %
                      (total_count / batch_size, total_loss / total_count, total_correct / total_count))

        accuracy = total_correct / total_count
        # tb.add_scalar('Loss/train', total_loss, epoch)
        # tb.add_scalar('Accuracy/train', accuracy, epoch)

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f, accuracy: %.3f" %
                  (epoch, init_epoch + num_epochs - 1, total_loss, accuracy))

        if save_every is not None \
                and epoch % save_every == 0 \
                and save_path is not None:
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
            }, save_path)
            print("Checkpoint saved: " + save_path)
            saved_epoch = True

    if save_path is not None and not saved_epoch:
        torch.save({
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
        }, save_path)
        print("Checkpoint saved: " + save_path)

    print("Finished Training")


def learn_stats(stats_net, data_loader, num_epochs=1):

    stats_net.start_tracking_stats()

    batch_total = 1

    print("Beginning tracking stats.")

    for epoch in range(1, num_epochs + 1):

        for batch_i, data in enumerate(data_loader, batch_total):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            stats_net(x)

        batch_total = batch_i + 1

    print("Finished Tracking Stats")


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
    if colorbar:
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
