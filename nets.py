import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def predict(self, x):
        self.eval()
        y = self.__call__(x)
        return torch.argmax(y, dim=-1)


class ResNet(Net):
    def __init__(self, num_classes, block_depth):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block_depth[0], 16, stride=1)
        self.layer2 = self._make_layer(block_depth[1], 32, stride=2)
        self.layer3 = self._make_layer(block_depth[2], 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

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

    def _make_layer(self, depth, num_filters, stride=1):
        layers = []
        for i in range(depth):
            if i != 0:
                stride = 1
            layers.append(ResidualBlock(self.in_channels, num_filters, stride))
            self.in_channels = num_filters

        return nn.Sequential(*layers)


class ResNet20(ResNet):
    def __init__(self, num_classes):
        super().__init__(num_classes, [3, 3, 3])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
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

    def __init__(self, layer_dims):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x):
        L = len(self.layers)
        for i in range(L):
            x = self.layers[i](x)
            if i < L - 1:
                x = F.relu(x)
        return x


class ConvNet(Net):

    def __init__(self, layer_dims, kernel_size):
        super().__init__()

        self.layers = nn.ModuleList()

        L = len(layer_dims) - 2
        in_channels = 1

        padding = (kernel_size - 1) // 2

        for i in range(L):
            out_channels = layer_dims[i]
            self.layers.append(nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=kernel_size,
                                         padding=padding))
            in_channels = out_channels
        self.layers.append(nn.Linear(layer_dims[i + 1],
                                     layer_dims[i + 2]))

    def forward(self, x):
        L = len(self.layers)
        for i in range(L):
            if i < L - 1:
                x = self.layers[i](x)
                x = F.relu(x)
            else:
                x = nn.Flatten()(x)
                x = self.layers[i](x)
        return x
