# import numpy as np
import torch
from functools import reduce


# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

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
    feature_dim = 1
    all_dims = list(range(len(data.shape)))
    # used in iteration over batches, skip channels
    dims_collapse = all_dims[1:-1]
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
    var = (S_2 - S**2 / n / weight) / n / weight
    nan_to_zero(mean)
    nan_to_zero(var)
    return mean, var, n


class StatsRecorder:
    def __init__(self, n_classes, data=None, labels=None):
        self.n_classes = n_classes
        self.initialized = False

        if data is not None:
            n_features = data.shape[1]
            shape = (n_classes, n_features)

            self.mean = torch.zeros(shape)
            self.var = torch.zeros(shape)
            self.mean, self.var, self.n = c_mean_var(data, labels, shape)

            self.initialized = True

    def update(self, data, labels):
        if not self.initialized:
            self.__init__(self.n_classes, data, labels)
        else:
            shape = self.mean.shape
            new_mean, new_var, m = c_mean_var(data, labels, shape)
            old_mean, old_var, n = self.mean, self.var, self.n

            # assert new_mean[m.squeeze() != 0].isfinite().all(), \
            #     "new_mean invalid before"
            # assert old_mean[n.squeeze() != 0].isfinite().all(), \
            #     "mean invalid before"

            self.mean, self.var, self.n = combine_mean_var(old_mean, old_var, n,
                                                           new_mean, new_var, m)

            # assert self.mean[self.n.squeeze() != 0].isfinite().all(), \
            #     "mean invalid after"


def batch_feature_mean_var(x, dim=1, unbiased=False):
    dims_collapse = list(range(len(x.shape)))
    dims_collapse.remove(dim)
    # dims_collapse = tuple(dims_collapse)
    mean = x.mean(dims_collapse)
    var = x.var(dims_collapse, unbiased=unbiased)
    return mean, var


# pylint: disable=no-member

torch.manual_seed(0)

n_classes = 10
n_features = 5
stats = StatsRecorder(n_classes)

data_shape = [n_features, 8, 2]
data = [torch.empty([0] + data_shape)] * n_classes

for i in range(300):
    n_samples = torch.randint(10, 101, size=(1,)).item()
    new_data = torch.randn(n_samples, *data_shape)
    new_labels = torch.randint(n_classes, size=(n_samples,))

    # print("incoming data shape: ", new_data.shape)

    for c in range(n_classes):
        data[c] = torch.cat((data[c], new_data[new_labels == c]))
        # print("data[{}].shape: ".format(c), data[c].shape)

    stats.update(new_data, new_labels)

    for c in range(n_classes):
        class_mean, class_var = batch_feature_mean_var(data[c])

        assert stats.mean[stats.n.squeeze() != 0].isfinite().all(), \
            "recorded mean has invalid entries"
        assert stats.mean[c].shape == class_mean.shape
        assert torch.allclose(stats.mean[c], class_mean, atol=1e-7, equal_nan=True), \
            "class {}".format(c) \
            + "\nclass mean: {}".format(class_mean) \
            + "\nrecorded mean: {}".format(stats.mean[c]) \
            + "\nerror: {}".format(torch.norm(stats.mean[c] - class_mean))

        assert stats.var[c].shape == class_var.shape
        assert stats.var[stats.n.squeeze() != 0].isfinite().all(), \
            "recorded var has invalid entries"
        assert torch.allclose(stats.var[c], class_var, equal_nan=True), \
            "class {}".format(c) \
            + "\nclass var: {}".format(class_var) \
            + "\nrecorded var: {}".format(stats.var[c]) \
            + "\nerror: {}".format(torch.norm(stats.var[c] - class_var))

    # print("assert {} passed".format(i))


mean, var = torch.empty_like(stats.mean), torch.empty_like(stats.mean)
for c in range(n_classes):
    mean[c], var[c] = batch_feature_mean_var(data[c])

print("cond mean error: ", torch.norm(stats.mean - mean))
print("cond var error: ", torch.norm(stats.var - var))

mean, var, _ = reduce_mean_var(stats.mean, stats.var, stats.n)
data = torch.cat(data)
true_mean, true_var = batch_feature_mean_var(data)
print("reduced mean error: ", torch.norm(mean - true_mean))
print("reduced var error: ", torch.norm(var - true_var))
