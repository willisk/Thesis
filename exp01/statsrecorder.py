import numpy as np
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


def c_mean_var(data, labels, shape):
    feature_dim = 1
    all_dims = list(range(len(data.shape)))
    dims_collapse = tuple(all_dims[1:-1])
    # count of collapsed dims
    weight = np.multiply(*data.shape[2:])  # XXX multiply dims collapse
    S, S_2 = np.zeros(shape), np.zeros(shape)
    n = np.zeros(shape[0])
    for d, c in zip(data, labels):
        S[c] += d.sum(dims_collapse)
        S_2[c] += (d**2).sum(dims_collapse)
        n[c] += 1
    n = expand_as_r(n, S)
    mean = S / n / weight
    var = (S_2 - S**2 / n / weight) / n / weight
    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)
    return mean, var, n


class StatsRecorder:
    def __init__(self, n_classes, data=None, labels=None):
        self.n_classes = n_classes
        self.initialized = False

        if data is not None:
            n_features = data.shape[1]
            shape = (n_classes, n_features)

            self.mean = np.zeros(shape)
            self.var = np.zeros(shape)
            self.mean, self.var, self.n = c_mean_var(data, labels, shape)

            self.initialized = True

    def update(self, data, labels):
        if not self.initialized:
            self.__init__(self.n_classes, data, labels)
        else:
            shape = self.mean.shape
            new_mean, new_var, m = c_mean_var(data, labels, shape)
            old_mean, old_var, n = self.mean, self.var, self.n

            self.mean, self.var, self.n = combine_mean_var(old_mean, old_var, n,
                                                           new_mean, new_var, m)


def batch_feature_mean_var(x, dim=1, unbiased=False):
    dims_collapse = list(range(len(x.shape)))
    dims_collapse.remove(dim)
    dims_collapse = tuple(dims_collapse)
    mean = x.mean(dims_collapse)
    var = x.var(dims_collapse)
    # XXX TUPLE, unbiased
    # var = x.var(dims_collapse, unbiased=unbiased)
    return mean, var


# pylint: disable=no-member
rs = np.random.RandomState(323)

n_classes = 10
n_features = 5
stats = StatsRecorder(n_classes)

data_shape = [n_features, 8, 2]
data = [np.empty([0] + data_shape)] * n_classes

for i in range(300):
    n_samples = rs.randint(10, 101)
    new_data = rs.randn(n_samples, *data_shape)
    new_labels = rs.randint(n_classes, size=n_samples)

    for c in range(n_classes):
        data[c] = np.vstack((data[c], new_data[new_labels == c]))

    stats.update(new_data, new_labels)

    for c in range(n_classes):
        class_mean, class_var = batch_feature_mean_var(data[c])
        assert stats.mean[c].shape == class_mean.shape
        assert stats.var[c].shape == class_var.shape
        assert np.allclose(stats.mean[c], class_mean)
        # assert np.allclose(stats.var[c], class_var)


mean, var = np.empty_like(stats.mean), np.empty_like(stats.mean)
for c in range(n_classes):
    mean[c], var[c] = batch_feature_mean_var(data[c])

print("cond mean error: ", np.linalg.norm(stats.mean - mean))
print("cond std error: ", np.linalg.norm(stats.var - var))

mean, var, _ = reduce_mean_var(stats.mean, stats.var, stats.n)
data = np.vstack(data)
print("mean error: ", np.linalg.norm(mean - data.mean((0, 2, 3))))
print("std error: ", np.linalg.norm(var - data.var((0, 2, 3))))
