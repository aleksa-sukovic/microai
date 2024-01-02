from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_labels: Optional[int] = None, z_dim: int = 100, img_shape: Tuple[int] = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.y_emb = nn.Embedding(n_labels, n_labels) if n_labels else None
        self.in_dim = z_dim + n_labels if n_labels else z_dim
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, int(np.prod(img_shape))), nn.Sigmoid(),
        )

    def forward(self, z, y):
        input = torch.cat((z, self.y_emb(y)), dim=-1) if self.y_emb else z
        img = self.model(input)
        return img.view(-1, *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, n_labels: Optional[int] = None, img_shape: Tuple[int] = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.y_emb = nn.Embedding(n_labels, n_labels) if n_labels else None
        self.in_dim = n_labels + int(np.prod(img_shape)) if n_labels else int(np.prod(img_shape))
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.Dropout(0.4), nn.ReLU(),
            nn.Linear(512, 512), nn.Dropout(0.4), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, y=None):
        input = x.view(x.size(0), -1)
        input = torch.cat((input, self.y_emb(y)), dim=-1) if y is not None else input
        return self.model(input)
