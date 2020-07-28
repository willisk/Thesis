import numpy as np
import matplotlib.pyplot as plt

import torch


def count_correct(outputs, labels):
    preds = outputs.argmax(dim=-1)
    return (preds == labels).sum()


def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, var, cc,
                   class_conditional=True, bessel_correction=True):

    if class_conditional:
        total = torch.zeros((n_classes, n_features))
        total.index_add_(0, labels, inputs)
        N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)
    else:
        total = inputs.sum(dim=0)
        N_class = len(inputs)

    cc_f = cc.float()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)
    ##
    total_2 = torch.zeros((n_classes, n_features))
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
