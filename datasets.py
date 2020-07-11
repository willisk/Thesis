import numpy as np
from sklearn.datasets import make_blobs, make_circles
# import matplotlib.pyplot as plt

import torch


class Dataset2D(torch.utils.data.Dataset):

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def generator(self, **params):
        return torch.utils.data.DataLoader(self, **params)

    def full(self):
        return self.X, self.Y


class Dataset01(Dataset2D):
    def __init__(self, **kwargs):
        X, Y = make_blobs(n_features=2, centers=3, **kwargs)
        X = np.array(X, dtype='float32')
        super(Dataset01, self).__init__(X, Y)


class Dataset02(Dataset2D):
    def __init__(self, **kwargs):
        X, Y = make_circles(**kwargs)
        X = np.array(X, dtype='float32')
        super(Dataset02, self).__init__(X, Y)

# dataset01_generator()
