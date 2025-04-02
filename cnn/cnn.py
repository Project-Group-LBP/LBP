import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(CNN, self).__init__()
        # Convolutional layers for 80x80 input
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(64)

        # Calculate feature dimension for 80x80 input
        self.feature_dim = self._calculate_conv_output_dim()

    def _calculate_conv_output_dim(self, input_dim=80):
        """Calculate output dimension after CNN layers"""
        dim = input_dim
        for _ in range(3):  # 3 conv layers
            dim = (dim - 3) // 2 + 1
        return 64 * dim * dim  # 64 channels * height * width

    def forward(self, x):
        # Handle input format conversion if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Standardize dimensions
        if len(x.shape) == 3:  # (height, width, channels)
            x = x.permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)
        elif len(x.shape) == 4 and x.shape[3] == 3:  # (batch, height, width, channels)
            x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)

        # Ensure x has the right dtype
        if x.dtype != torch.float32:
            x = x.float()

        # Forward pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = x.reshape(x.size(0), -1)  # Flatten with shape compatibility
        return x
