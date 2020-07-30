import numpy as np
import matplotlib.pyplot as plt

import torch

import itertools


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum()


def expand_as_r(a, b):
    diff = len(b.shape) - len(a.shape)
    shape = list(a.shape) + diff * [1]
    return a.reshape(shape)


def cat_cond_mean_(inputs, labels, mean, var, cc,
                   class_conditional=True, bessel_correction=True):

    shape = mean.shape
    n_classes = shape[0]

    if class_conditional:
        total = torch.zeros(shape)
        total.index_add_(0, labels, inputs)
        N_class = expand_as_r(torch.bincount(labels, minlength=n_classes), cc)
    else:
        total = inputs.sum(dim=0)
        N_class = len(inputs)

    cc_f = cc.float()
    cc.add_(N_class)

    mean.mul_(cc_f / cc)
    mean.add_((total / cc).expand_as(mean))
    ##
    total_2 = torch.zeros(shape)
    # total_var.index_add_(0, labels, inputs - mean[labels])
    for i, x in enumerate(inputs):
        total_2[labels[i]] += (inputs[i])**2
    var.mul_(cc_f / cc)
    var.add_((total_2 - N_class * mean**2) / cc)


def train(net, data_loader, criterion, optimizer, num_epochs=1, print_every=10):
    "Training Loop"

    net.train()

    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for data in data_loader:
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = len(inputs)
            total_count += batch_size
            total_loss += loss.item() * batch_size
            total_correct += count_correct(outputs, labels)

        accuracy = total_correct / total_count
        # tb.add_scalar('Loss', total_loss, epoch)
        # tb.add_scalar('Accuracy', accuracy, epoch)

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, total_loss))
    print("Finished Training")


def learn_stats(stats_net, data_loader, num_epochs=1):

    stats_net.start_tracking_stats()

    batch_total = 1

    for epoch in range(1, num_epochs + 1):

        total_count = 0.0
        total_loss = 0.0
        total_correct = 0.0

        for batch_i, data in enumerate(data_loader, batch_total):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}
            stats_net(x)

            # for i, layer in enumerate(stats_net.stats, 1):
            #     for j, feature in enumerate(layer.running_mean):
            #         for m, f in enumerate(feature):
            #             tb.add_scalar("stats%i.class%i.f%i" % (i, j, m + 1),
            #                           feature[m], batch_i)

        batch_total = batch_i + 1

    stats_net.disable_hooks()
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


def plot_prediction2d(data, labels, net, num=400, axis=None, cmap='Spectral', contourgrad=False):

    print(data.shape)
    X, Y = data, labels
    h = 0.05
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))
    mesh = (np.c_[xx.ravel(), yy.ravel()])
    mesh = torch.from_numpy(mesh.astype('float32'))
    Z = net.predict(mesh)
    Z = Z.T.reshape(xx.shape)

    if axis is None:
        _plt = plt
        plt.figure(figsize=(5, 5))
    else:
        _plt = axis

    # if contourgrad:
    #     A = net(mesh)
    #     _plt.contourf(xx, yy, A.T.reshape(xx.shape), cmap=cmap, alpha=.3)
    # else:
    _plt.contourf(xx, yy, Z, cmap=cmap, alpha=.3)
    _plt.contour(xx, yy, Z, colors='k')
    _plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap=cmap, alpha=.4)


def plot_num_matrix(X, labels, title_fmt, colors=None):
    L = len(X)
    ncols = 3
    nrows = -(-L // ncols)
    for i in range(L):
        plt.subplot(nrows, ncols, i + 1)
        plt.tight_layout()
        plt.imshow(X[i].squeeze(),
                   cmap='gray', interpolation='none')
        if colors is None:
            color = 'k'
        else:
            color = colors[i]
        plt.title(title_fmt.format(labels[i]), color=color)
        plt.xticks([])
        plt.yticks([])
