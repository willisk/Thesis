import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from torchvision.models import resnet34 as _resnet34, resnet50 as ResNet50
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
# import torchvision.models as models
import segmentation_models_pytorch as smp


class DistortionModelAffine(nn.Module):
    def __init__(self, lambd, n_dims):
        super().__init__()
        self.noise = nn.Parameter(
            torch.randn((n_dims)).unsqueeze(0)) * lambd
        self.linear = (torch.eye(n_dims) + torch.randn((n_dims, n_dims)) * lambd)

    @torch.no_grad()
    def forward(self, inputs):
        outputs = inputs + self.noise
        return outputs @ self.linear


class DistortionModelConv(nn.Module):
    def __init__(self, lambd, input_shape):
        super().__init__()

        kernel_size = 3
        nch = input_shape[0]

        self.conv1 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(nch, nch, kernel_size,
                               padding=1, padding_mode='reflect')
        self.noise = nn.Parameter(torch.randn(input_shape).unsqueeze(0))

        self.conv1.weight.data.normal_()
        self.conv2.weight.data.normal_()
        self.conv1.weight.data *= lambd
        self.conv2.weight.data *= lambd
        for f in range(nch):
            self.conv1.weight.data[f][f][1][1] += 1
            self.conv2.weight.data[f][f][1][1] += 1

        self.conv1.bias.data.normal_()
        self.conv2.bias.data.normal_()
        self.conv1.bias.data *= lambd
        self.conv2.bias.data *= lambd

        self.noise.data *= lambd

    @torch.no_grad()
    def forward(self, inputs):
        outputs = inputs
        outputs = outputs + self.noise
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


def conv1x1Id(n_chan):
    conv = nn.Conv2d(n_chan, n_chan,
                     kernel_size=1,
                     bias=True,
                     )
    conv.weight.data.fill_(0)
    for i in range(n_chan):
        conv.weight.data[i, i, 0, 0] = 1
    return conv


class ReconstructionModelAffine(nn.Module):
    def __init__(self, args, input_shape, n_dims, n_classes):
        super().__init__()

        self.bias = nn.Parameter(
            torch.zeros((n_dims)).unsqueeze(0))
        self.linear = nn.Parameter(
            torch.eye(n_dims) + torch.randn((n_dims, n_dims)) / np.sqrt(n_dims))

    def forward(self, inputs):
        return inputs @ self.linear + self.bias


class ReconstructionModelResnet(nn.Module):
    def __init__(self, args, input_shape, n_dims, n_classes, bias=True):
        super().__init__()

        n_chan = input_shape[0]
        self.conv1x1 = conv1x1Id(n_chan)
        self.bn = nn.BatchNorm2d(n_chan)

        self.invert_block = nn.Sequential(*[
            InvertBlock(
                n_chan,
                args.r_block_width,
                noise_level=1 / np.sqrt(n + 1),
                relu_out=n < args.r_block_depth - 1,
                bias=bias,
            ) for n in range(args.r_block_depth)
        ])

    def forward(self, inputs):
        outputs = self.conv1x1(inputs)
        outputs = self.invert_block(outputs)
        outputs = self.bn(outputs)
        return outputs


class ReconstructionModelUnet(smp.Unet):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
        )


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return torch.argmax(self(x), dim=1)


class tResNet(Net):
    def __init__(self, in_channels, n_classes, block_depth):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 16, block_depth[0], stride=1)
        self.layer2 = self._make_layer(16, 32, block_depth[1], stride=2)
        self.layer3 = self._make_layer(32, 64, block_depth[2], stride=2)

        fc_dim = max([d for d, n in zip([16, 32, 64], block_depth) if n > 0])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(fc_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, in_channels, num_filters, depth, stride=1):
        layers = []
        for i in range(depth):
            if i != 0:
                stride = 1
            layers.append(ResidualBlock(in_channels, num_filters, stride))
            in_channels = num_filters

        return nn.Sequential(*layers)


class ResNet9(tResNet):
    def __init__(self, in_channels, n_classes):
        super().__init__(in_channels, n_classes, [2, 2, 0])


class ResNet20(tResNet):
    def __init__(self, in_channels, n_classes):
        super().__init__(in_channels, n_classes, [3, 3, 3])


class InvertBlock(nn.Module):
    def __init__(self, n_channels, n_features, noise_level=0.1,
                 relu_out=True, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, n_features,
                               kernel_size=3,
                               padding=1,
                               bias=bias,
                               padding_mode='reflect')

        self.bn1 = nn.BatchNorm2d(n_features)

        self.conv2 = nn.Conv2d(n_features, n_channels,
                               kernel_size=3,
                               padding=1,
                               bias=bias,
                               padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU()

        self.conv1.weight.data *= noise_level
        self.conv2.weight.data *= noise_level

        if bias:
            self.conv1.bias.data *= 0
            self.conv2.bias.data *= 0

        self.relu_out = relu_out

    def forward(self, x):

        y = x

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + y

        if self.relu_out:
            x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False, padding_mode='zeros'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connection = nn.Sequential()

    def forward(self, x):

        y = x

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + self.skip_connection(y)

        x = self.relu(x)

        return x


class FCNet(Net):

    def __init__(self, layer_dims, batch_norm=False):
        super().__init__()

        layers = []
        L = len(layer_dims) - 1
        for i in range(L):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            if i < L - 1:
                layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ConvNet(Net):

    def __init__(self, input_dims, layer_dims, out_dims, kernel_size):
        super().__init__()

        self.layers = nn.ModuleList()

        padding = (kernel_size - 1) // 2

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Conv2d(layer_dims[i],
                                         layer_dims[i + 1],
                                         kernel_size=kernel_size,
                                         padding=padding))
        self.layers.append(nn.Linear(input_dims * layer_dims[i + 1],
                                     out_dims))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.layers[-1](x)
        return x


def resnet18():
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    resnet.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return resnet


def resnet34():
    resnet = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
    resnet.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return resnet


def resnet50():
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    resnet.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    return resnet
