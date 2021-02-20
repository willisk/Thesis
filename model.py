import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNET(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            features=[64, 128, 256, 512],
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for n_features in features:
            self.encoder.append(ConvBlock(in_channels, n_features))
            in_channels = n_features

        # Decoder
        for n_features in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(
                    n_features * 2, n_features, kernel_size=2, stride=2,
                )
            )
            self.decoder.append(ConvBlock(n_features * 2, n_features))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = list(reversed(skip_connections))

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)


if __name__ == "__main__":
    x = torch.randn((8, 3, 160, 160))
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
