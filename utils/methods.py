import os
import sys
from . import utility

import numpy as np

import torch
import torch.nn.functional as F

USE_DRIVE = True


def get_methods(DATA_A, net, dataset, args, DEVICE):

    MODELDIR = dataset.data_dir
    n_dims = dataset.n_dims
    n_classes = dataset.n_classes

    # ======= Setup Methods =======
    STD = args.use_std
    stats_path = os.path.join(MODELDIR, "stats_{}.pt")

    # Random Projections
    n_random_projections = args.n_random_projections

    # ======= NN Project =======
    criterion = torch.nn.CrossEntropyLoss()

    # NOTE: when using bn_layers, use inputs from hook
    # net_layers = utility.get_bn_layers(net)
    net_layers = utility.get_child_modules(net)[:-1]
    layer_activations = [None] * len(net_layers)
    net_last_outputs = None

    def layer_hook_wrapper(activations, idx):
        def hook(_module, _inputs, outputs):
            activations[idx] = outputs
        return hook

    for l, layer in enumerate(net_layers):
        layer.register_forward_hook(layer_hook_wrapper(layer_activations, l))

    def project_NN(data):
        nonlocal net_last_outputs
        inputs, labels = data
        net_last_outputs = net(inputs)
        return layer_activations[-1]

    def project_NN_all(data):
        nonlocal net_last_outputs
        inputs, labels = data
        net_last_outputs = net(inputs)
        return [inputs] + layer_activations

    # ======= Random Net =======

    random_net_path, random_net = dataset.net()
    random_net_path = '_init.'.join(random_net_path.split('.'))

    @utility.store_data
    def random_net_state_dict():
        # _, random_net = dataset.net()
        random_net.to(DEVICE)
        random_net.train()

        with torch.no_grad():
            for inputs, labels in DATA_A:
                random_net(inputs)

        return random_net.state_dict()

    state_dict = random_net_state_dict(  # pylint: disable=unexpected-keyword-arg
        path=random_net_path, map_location=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)
    random_net.load_state_dict(state_dict)
    random_net.to(DEVICE)
    random_net.eval()

    random_layer_activations = []

    for l, layer in enumerate(utility.get_child_modules(random_net)[:-1]):
        random_layer_activations.append(None)
        layer.register_forward_hook(
            layer_hook_wrapper(random_layer_activations, l))

    def project_random_NN_all(data):
        inputs, labels = data
        random_net(inputs)
        return [inputs] + random_layer_activations

    # ======= Random Projections =======
    rp_hash = f"{n_random_projections}"
    path_RP = os.path.join(MODELDIR, f"RP-{rp_hash}")

    @utility.store_data
    def random_projections():
        RP = torch.randn((n_dims, n_random_projections)).to(DEVICE)
        RP = RP / RP.norm(2, dim=0)
        return RP

    # for reproducibility
    RP = random_projections(  # pylint: disable=unexpected-keyword-arg
        path=path_RP, map_location=DEVICE, use_drive=USE_DRIVE)

    def get_input(data):
        return data[0]

    mean_A, std_A = utility.collect_stats(
        DATA_A, get_input, n_classes, class_conditional=False, std=True, keepdim=True,
        path=stats_path.format('inputs'), device=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)
    mean_A_C, std_A_C = utility.collect_stats(
        DATA_A, get_input, n_classes, class_conditional=True, std=True, keepdim=True,
        path=stats_path.format('inputs-CC'), device=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)

    # min_A, max_A = utility.collect_min_max(
    #     DATA_A, path=stats_path.format('min-max'), device=DEVICE, use_drive=USE_DRIVE)

    def project_RP(data):
        X, Y = data
        return (X - mean_A).reshape(-1, n_dims) @ RP

    def project_RP_CC(data):
        X, Y = data
        X_proj_C = None
        for c in range(n_classes):
            mask = Y == c
            X_proj_c = (X[mask] - mean_A_C[c]).reshape(-1, n_dims) @ RP
            if X_proj_C is None:
                X_proj_C = torch.empty((X.shape[0], n_random_projections),
                                       dtype=X_proj_c.dtype, device=X.device)
            X_proj_C[mask] = X_proj_c
        return X_proj_C

    mean_RP_A, std_RP_A = utility.collect_stats(
        DATA_A, project_RP, n_classes, class_conditional=False, std=True, keepdim=True,
        path=stats_path.format(f"RP-{rp_hash}"), device=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)
    mean_RP_A_C, std_RP_A_C = utility.collect_stats(
        DATA_A, project_RP_CC, n_classes, class_conditional=True, std=True, keepdim=True,
        path=stats_path.format(f"RP-CC-{rp_hash}"), device=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)

    # Random ReLU Projections
    f_rp_relu = 1 / 2
    relu_bias = mean_RP_A + f_rp_relu * std_RP_A * torch.randn_like(mean_RP_A)
    relu_bias_C = (mean_RP_A_C +
                   f_rp_relu * std_RP_A_C * torch.randn_like(mean_RP_A_C)).squeeze()

    def project_RP_relu(data):
        return F.relu(project_RP(data) + relu_bias)

    def project_RP_relu_CC(data):
        X, Y = data
        return F.relu(project_RP_CC(data) + relu_bias_C[Y])

    # ======= Combined =======

    def combine(project1, project2):
        def _combined_fn(data):
            out1 = project1(data)
            out2 = project2(data)
            if not isinstance(out1, list):
                out1 = [out1]
            if not isinstance(out2, list):
                out2 = [out2]
            return out1 + out2
        return _combined_fn

    # ======= Loss Function =======
    f_crit = args.f_crit
    f_reg = args.f_reg
    f_stats = args.f_stats

    def regularization(x):
        diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]
        diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]
        diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]
        return (
            torch.norm(diff1.reshape(len(x), -1), dim=1)
            + torch.norm(diff2.reshape(len(x), -1), dim=1)
            + torch.norm(diff3.reshape(len(x), -1), dim=1)
            + torch.norm(diff4.reshape(len(x), -1), dim=1)
        ).mean()  # / (x.prod)

    # @debug

    def loss_stats(stats_a, stats_b, class_conditional=False):
        if not isinstance(stats_a, list):
            stats_a, stats_b = [stats_a], [stats_b]
        assert len(stats_a) == len(stats_b), "lists need to be of same length"
        num_maps = len(stats_a)
        loss = torch.tensor(0).float().to(DEVICE)
        info = {}
        for i, ((ma, sa), (mb, sb)) in enumerate(zip(stats_a, stats_b)):
            ma, sa = ma.squeeze(), sa.squeeze()
            mb, sb = mb.squeeze(), sb.squeeze()
            if not class_conditional:
                loss_m = (ma - mb).norm()
                loss_s = (sa - sb).norm()
            else:   # class conditional
                if ma.ndim == 1:
                    loss_m = (ma - mb).abs().mean()
                    loss_s = (sa - sb).abs().mean()
                else:  # multiple features
                    loss_m = (ma - mb).norm(dim=1).mean()
                    loss_s = (sa - sb).norm(dim=1).mean()
            loss_m /= num_maps
            loss_s /= num_maps
            if num_maps > 1:
                info[f'[statistics losses means] {i}'] = loss_m.item()
                info[f'[statistics losses vars] {i}'] = loss_s.item()
            else:
                info['[statistics losses] mean'] = loss_m.item()
                info['[statistics losses] var'] = loss_s.item()
            if loss_m.isfinite():   # mean, variance is nan if batch empty
                loss += loss_m
            if loss_s.isfinite():
                loss += loss_s
        info['[losses] statistics'] = loss.item()
        return loss, info

    def loss_fn_wrapper(name, project, class_conditional, f_stats_scale=1):
        _name = name.replace(' ', '-')
        if "RP" in _name or "COMBINED" in _name:
            _name = f"{_name}-{rp_hash}"

        stats_A = utility.collect_stats(
            DATA_A, project, n_classes, class_conditional,
            std=STD, path=stats_path.format(_name), device=DEVICE, use_drive=USE_DRIVE, reset=args.reset_stats)

        def _loss_fn(data, project=project, class_conditional=class_conditional, f_stats_scale=f_stats_scale):
            nonlocal net_last_outputs
            net_last_outputs = None

            inputs, labels = data

            info = {}
            loss = torch.tensor(0).float().to(DEVICE)

            if f_reg:
                loss_reg = f_reg * regularization(inputs)
                info['[losses] regularization'] = loss_reg.item()
                loss += loss_reg

            if f_stats:
                outputs = project(data)
                stats = utility.get_stats(
                    outputs, labels, n_classes, class_conditional=class_conditional, std=STD)
                cost_stats, info_stats = loss_stats(
                    stats_A, stats, class_conditional=class_conditional)
                for k, v in info_stats.items():
                    info[k] = f_stats * f_stats_scale * v
                loss += f_stats * f_stats_scale * cost_stats

            if net_last_outputs is None:
                net_last_outputs = net(inputs)

            info['accuracy'] = utility.count_correct(
                net_last_outputs, labels) / len(labels)

            if f_crit:
                loss_crit = f_crit * criterion(net_last_outputs, labels)
                info['[losses] criterion'] = loss_crit.item()
                loss += loss_crit

            info['loss'] = loss
            return info
            # return loss
        return name, _loss_fn

    def criterion_only(data):
        inputs, labels = data
        outputs = net(inputs)

        info = {}

        loss_crit = f_crit * criterion(outputs, labels)

        if f_reg:
            loss_reg = f_reg * regularization(inputs)
            info['[losses] regularization'] = loss_reg.item()
            info['[losses] criterion'] = loss_crit.item()
            loss = loss_crit + loss_reg
        else:
            loss = loss_crit

        info['loss'] = loss
        info['accuracy'] = utility.count_correct(outputs, labels) / len(labels)

        return info

    methods = [
        ("CRITERION", criterion_only),
        loss_fn_wrapper(
            name="NN",
            project=project_NN,
            class_conditional=False,
        ),
        loss_fn_wrapper(
            name="NN CC",
            project=project_NN,
            class_conditional=True,
        ),
        loss_fn_wrapper(
            name="NN ALL",
            project=project_NN_all,
            class_conditional=False,
        ),
        loss_fn_wrapper(
            name="NN ALL CC",
            project=project_NN_all,
            class_conditional=True,
        ),
        loss_fn_wrapper(
            name="RANDOM NN",
            project=project_random_NN_all,
            class_conditional=False,
        ),
        loss_fn_wrapper(
            name="RANDOM NN CC",
            project=project_random_NN_all,
            class_conditional=True,
        ),
        loss_fn_wrapper(
            name="RP",
            project=project_RP,
            class_conditional=False,
        ),
        loss_fn_wrapper(
            name="RP CC",
            project=project_RP_CC,
            class_conditional=True,
        ),
        loss_fn_wrapper(
            name="RP ReLU",
            project=project_RP_relu,
            class_conditional=False,
        ),
        loss_fn_wrapper(
            name="RP ReLU CC",
            project=project_RP_relu_CC,
            class_conditional=True,
        ),
        loss_fn_wrapper(
            name="COMBINED CC",
            project=combine(project_NN_all, project_RP_CC),
            class_conditional=True,
        ),
    ]

    return methods
