import numpy as np
from sklearn.datasets import make_blobs, make_circles
# import matplotlib.pyplot as plt

import torch


class Dataset2D(torch.utils.data.Dataset):

    def __init__(self, type, **params):
        if type == 0:
            X, Y = make_blobs(n_features=2, centers=3, **params)
            X = np.array(X, dtype='float32')
        if type == 1:
            X, Y = make_circles(**params)
            X = np.array(X, dtype='float32')
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def generator(self, **params):
        return torch.utils.data.DataLoader(self, **params)

    def full(self):
        return self.X, self.Y
