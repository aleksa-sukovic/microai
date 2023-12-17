from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.projection = conv1x1(in_channels, out_channels, stride) if in_channels != out_channels else None
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)

    def forward(self, x):
        residual = self.projection(x) if self.projection else x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.relu(x + residual)


class ResNet(nn.Module):
    def __init__(self, num_classes: int, groups: List[int], init_filters: int = 64, preprocess: Optional[nn.Module] = None):
        super().__init__()
        self.num_filters = init_filters
        self.conv = preprocess or nn.Conv2d(3, self.num_filters, kernel_size=7, stride=2, padding=3)
        self.batch_norm = nn.BatchNorm2d(self.num_filters)
        self.max_pool = None if preprocess else nn.MaxPool2d(kernel_size=3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # dynamically selects kernel and stride to obtain the desired size
        self.fc = nn.Linear(
            in_features=self.num_filters * 2**(len(groups) - 1), # each layer group doubles the number of filters
            out_features=num_classes,
        )

        groups_modules = [
            ResNetBlock(self.num_filters, self.num_filters, stride=1) for _ in range(groups[0])] # first group does not shrink the input

        for i in range(1, len(groups)):
            curr_group_channels = self.num_filters * 2**i
            prev_group_channels = self.num_filters * 2**(i-1)
            groups_modules += [
                ResNetBlock(
                    in_channels=prev_group_channels if j == 0 else curr_group_channels, # first block in the group shrinks the input
                    stride=2 if j == 0 else 1,
                    out_channels=curr_group_channels, # remaining blocks do not shrink the input
                ) for j in range(groups[i])]

        self.groups = nn.Sequential(*groups_modules)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv(x)))
        x = self.max_pool(x) if self.max_pool else x
        x = self.groups(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
