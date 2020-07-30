# %%
import numpy as np
import torch
import timeit


## iteration + bincount
def cat_cond_mean_(inputs, labels, n_classes, n_features, mean, cc):
    batch_size = len(inputs)
    total = torch.zeros((n_classes, n_features))
    for i, sample in enumerate(inputs):
        total[labels[i]] += sample

    N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)
    cc_f = cc.float()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)

## mask + bincount


def cat_cond_mean2_(inputs, labels, n_classes, n_features, mean, cc):

    total = torch.zeros((n_classes, n_features))
    total.index_add_(0, labels, inputs.detach())

    N_class = torch.bincount(labels, minlength=n_classes).unsqueeze(-1)

    cc_f = cc.clone()
    cc.add_(N_class)
    mean.mul_(cc_f / cc)
    mean.add_(total / cc)


num_classes = 1000
num_features = 128 * 128
batch_size = 64

x = torch.rand((batch_size, num_features))
y = torch.randint(size=(batch_size, 1), high=num_classes).squeeze()


def verify():
    mean1 = torch.zeros((num_classes, num_features))
    cc1 = torch.ones((num_classes, 1))
    mean2 = torch.zeros((num_classes, num_features))
    cc2 = torch.ones((num_classes, 1))

    cat_cond_mean_(x, y, num_classes, num_features, mean1, cc1)
    cat_cond_mean2_(x, y, num_classes, num_features, mean2, cc2)

    print("allclose mean: " + str(torch.allclose(mean1, mean2)))
    print("allclose cc: " + str(torch.allclose(cc1, cc2)))


verify()


def test1():
    cat_cond_mean_(x, y, num_classes, num_features, mean, class_count)


def test2():
    cat_cond_mean2_(x, y, num_classes, num_features, mean, class_count)


mean = torch.zeros((num_classes, num_features))
class_count = torch.ones((num_classes, 1))
timeit.timeit(test1, number=100)

mean = torch.zeros((num_classes, num_features))
class_count = torch.ones((num_classes, 1))
timeit.timeit(test2, number=100)

print("done")
