import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

PWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PWD)

import datasets
import statsnet
import utility
import deepinversion
import shared

if sys.argv[0] == 'ipykernel_launcher':
    import importlib
    importlib.reload(datasets)
    importlib.reload(statsnet)
    importlib.reload(utility)
    importlib.reload(deepinversion)
    importlib.reload(shared)


# ======= Set Seeds =======
np.random.seed(0)
torch.manual_seed(0)

# ======= Setup Tensorboard =======
LOGDIR = os.path.join(PWD, "projectOK/runs")
LOGDIR, _ = utility.search_drive(LOGDIR)
shared.init_summary_writer(LOGDIR)
# writer = shared.get_summary_writer("main")

# ======= Hyperparameters =======
# hp = shared.parse_args_to_hyperparameters()
hp = dict(
    method='standard',
    cc=True,
    mask_bn=False,
    use_bn_stats=False,
    n_steps=100,
    learning_rate=0.01,
    factor_reg=0,
    factor_input=0,
    factor_layer=0,
    factor_criterion=1,
    distr_a=1,
    distr_b=1,
)
hp_string = utility.dict_to_str(hp)
# writer = shared.get_summary_writer(hp_string)
print("Hyperparameters:")
print(hp_string)

# ======= Create Dataset =======
dataset = datasets.DatasetGMM(
    n_dims=2, n_classes=3, n_modes=8,
    scale_mean=5, scale_cov=1,
    n_samples_per_class=int(1e3)
)

# dataset.plot()
# plt.show()
# print(dataset.JS())
stats_net = dataset.load_statsnet(
    # resume_training=True,
    use_drive=True,
)
dataset.print_accuracy(stats_net)

# dataset.plot(stats_net)
# dataset.plot_stats(stats_net)
# plt.show()

# cmaps = utility.categorical_cmaps(dataset.get_num_classes())
criterion = dataset.get_criterion()

# Create Dataset B, which is newly sampled, perturbed from distribution A
X_A, Y = dataset.sample(int(1e3))
X_A = torch.from_numpy(X_A)
Y = torch.LongTensor(Y)

utility.print_accuracy(X_A, Y, stats_net,
                       "net {accuracy} accuracy on unseen data A")


perturb_factor = 0.3
n_dims = X_A.shape[1]
shape = (n_dims, n_dims)
perturb_matrix = torch.eye(n_dims) + perturb_factor * torch.randn(shape)
perturb_shift = perturb_factor * torch.randn(n_dims)

X_B = X_A @ perturb_matrix + perturb_shift
# dataset.plot(data={'inputs': X_B, 'labels': Y}, net=stats_net)

utility.print_accuracy(X_B, Y, stats_net,
                       "net {accuracy} accuracy on data B")


A = torch.randn((n_dims, n_dims), requires_grad=True)
b = torch.randn((n_dims), requires_grad=True)


def preprocessing(X):
    return X @ A + b


# print("A", A)
optimizer = torch.optim.Adam([A, b], lr=0.1)
stats_net.init_hyperparameters(hp)
loss_fn = deepinversion.inversion_loss(stats_net, criterion, Y, hp)
# print("optim ", optimizer.param_groups)

# print("X_b[0]", X_B[0])
invert = deepinversion.deep_inversion(X_B,
                                      loss_fn,
                                      optimizer,
                                      steps=hp['n_steps'],
                                      pre_fn=preprocessing,
                                      )

# print("A", A)
utility.print_accuracy(preprocessing(X_B), Y, stats_net,
                       "net {accuracy} accuracy on data B")
