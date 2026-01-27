import torch.nn as nn


class SideBranch(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # output shape: [B, C, 1, 1]
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.gap(x)             # [B, C, 1, 1]
        x = x.view(x.size(0), -1)   # [B, C]
        x = self.fc(x)              # [B, num_classes]
        return x
