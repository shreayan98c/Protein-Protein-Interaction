import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, levels: int = 3, blocks: int = 1, channels: int = 32, dropout: float = 0.5):
        super().__init__()

        self.encoder = nn.ModuleList()

        prev_channels = 1
        for l in range(levels):
            for _ in range(blocks):
                output_channels = channels * 2 ** l
                self.encoder.append(nn.Sequential(
                    nn.Conv2d(prev_channels, output_channels, 3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                )
                )
                prev_channels = output_channels
            self.encoder.append(nn.MaxPool2d(2))
            self.encoder.append(nn.Dropout(dropout))

        self.encoder.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.encoder.append(nn.Flatten())

        self.head = nn.Sequential(
            nn.Linear(prev_channels, 10),
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.head(x)
        return x


class SimpleLinearModel(nn.Module):
    def __init__(self, hidden_layers: list = None, dropout: float = 0.3):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [50, 25, 3, 1]
        self.layers = nn.ModuleList()

        prev_channels = 10000
        for i, l in enumerate(hidden_layers[:-1]):
            self.layers.append(nn.Linear(prev_channels, l))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_channels = l

        self.layers.append(nn.Linear(prev_channels, 1))
        self.layers.append(nn.Sigmoid())

        # weight initialization
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                torch.nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
