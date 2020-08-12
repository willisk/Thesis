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
    S, S_2 = np.zeros(shape), np.zeros(shape)
    n = np.zeros(shape[0])
    for d, c in zip(data, labels):
        S[c] += d
        S_2[c] += d**2
        n[c] += 1
    n = expand_as_r(n, S)
    mean = S / n
    var = (S_2 - S**2 / n) / n
    mean = np.nan_to_num(mean)
    var = np.nan_to_num(var)
    return mean, var, n


class StatsRecorder:
    def __init__(self, n_classes, data=None, labels=None):
        self.n_classes = n_classes
        self.initialized = False

        if data is not None:
            shape = [self.n_classes] + list(data.shape)

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


# pylint: disable=no-member
rs = np.random.RandomState(323)

n_classes = 10
stats = StatsRecorder(n_classes)

data_shape = [4, 8, 2]
data = [np.empty([0] + data_shape)] * n_classes

for i in range(500):
    n_samples = rs.randint(10, 101)
    new_data = rs.randn(n_samples, *data_shape)
    new_labels = rs.randint(n_classes, size=n_samples)

    for c in range(n_classes):
        data[c] = np.vstack((data[c], new_data[new_labels == c]))

    stats.update(new_data, new_labels)

    for c in range(n_classes):
        assert np.allclose(stats.mean[c], data[c].mean(axis=0))
        assert np.allclose(np.sqrt(stats.var[c]), data[c].std(axis=0))

mean, std = np.empty_like(stats.mean), np.empty_like(stats.mean)
for c in range(n_classes):
    mean[c] = data[c].mean(axis=0)
    std[c] = data[c].std(axis=0)

print("cond mean error: ", np.linalg.norm(stats.mean - mean))
print("cond std error: ", np.linalg.norm(np.sqrt(stats.var) - std))

mean, var, _ = reduce_mean_var(stats.mean, stats.var, stats.n)
data = np.vstack(data)
print("mean error: ", np.linalg.norm(mean - data.mean(axis=0)))
print("std error: ", np.linalg.norm(np.sqrt(var) - data.std(axis=0)))
