"""Reusable neural network building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard residual block: conv -> BN -> ReLU -> conv -> BN + skip.

    Args:
        num_filters: Number of convolutional filters.
        kernel_size: Convolution kernel size.
    """

    def __init__(self, num_filters: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            num_filters, num_filters, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            num_filters, num_filters, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        result: torch.Tensor = self.relu(out)
        return result
