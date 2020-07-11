import numpy as np
import matplotlib.pyplot as plt

import torch


def train(net, data_loader, num_epochs=1, print_every=10):

    net.train()

    for epoch in range(1, num_epochs + 1):

        running_loss = 0.0

        for i, data in enumerate(data_loader, 1):
            inputs, labels = data
            x = {'inputs': inputs,
                 'labels': labels}

            net.optimizer.zero_grad()

            outputs = net(x)
            loss = net.criterion(outputs, labels)
            loss.backward()
            net.optimizer.step()

            running_loss += loss.item()

        if epoch % print_every == 0:
            print("[%d / %d] loss: %.3f" %
                  (epoch, num_epochs, running_loss))
    print("Finished Training")


def plot_decision_boundary(dataset, net, contourgrad=False):

    cmap = 'Spectral'
    X, Y = dataset.full()
    h = 0.05
    x_min, x_max = X[:, 0].min() - 10*h, X[:, 0].max() + 10*h
    y_min, y_max = X[:, 1].min() - 10*h, X[:, 1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = (np.c_[xx.ravel(), yy.ravel()])
    mesh = torch.from_numpy(mesh.astype('float32'))
    Z = net.predict(mesh)
    Z = Z.T.reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    if contourgrad:
        A = net(mesh)
        plt.contourf(xx, yy, A.T.reshape(xx.shape), cmap=cmap, alpha=.3)
    else:
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=.3)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y.squeeze(), cmap=cmap)
    # plt.scatter(np.mean(X[:, 0]), np.mean(X[:, 1]), c='g')


def plot_stats_mean(mean):
    cmap = 'Spectral'
    L = mean.shape[0]
    if L == 1:
        plt.scatter(mean[0], mean[1], c='k')
    else:
        mean = mean.T
        plt.scatter(mean[0], mean[1], c=np.arange(L),
                    cmap=cmap, edgecolors='k')


def cat_cond_mean(input, labels, n_classes, n_features, mean=None, N_class=None):
    batch_size = len(input)
    # mask = torch.zeros((batch_size, n_classes, n_features))
    # mask[torch.arange(batch_size), labels, torch.arange(n_features)] = 1
    # mask = F.one_hot(labels, num_classes=n_classes)  # (64, 3)
    # c = input.unsqueeze(1).repeat(1, n_classes, 1)

    if not N_class:
        N_class = torch.ones(n_classes).unsqueeze(-1)
    if not mean:
        mean = torch.zeros(n_classes, n_features)

    total = torch.zeros((batch_size, n_classes, n_features))
    for i, sample in enumerate(input):
        total[i, labels[i]] = sample

    mean = mean * N_class
    N_class += torch.bincount(labels, minlength=n_classes).unsqueeze(-1)
    mean = mean + total.sum(axis=0)
    mean = mean / N_class

    return mean, N_class
