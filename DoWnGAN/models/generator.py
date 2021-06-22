# Borrowed from: 
# https://github.com/Lornatang/SRGAN-PyTorch/
# Adopted to ESRGAN: https://arxiv.org/abs/1809.00219

import torch
import torch.nn as nn

class Generator(nn.Module):
    r"""The main architecture of the generator."""

    def __init__(self, coarse_dim, fine_dim, nc):
        r""" This is an esrgan model defined by the author himself.
        """
        super(Generator, self).__init__()
        # First layer.
        self.coarse_dim = coarse_dim
        self.fine_dim = fine_dim
        self.nc = nc
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.nc, self.coarse_dim, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks.
        residual_blocks = []
        for _ in range(16):
            residual_blocks.append(ResidualBlock(self.coarse_dim))
        self.Trunk = nn.Sequential(*residual_blocks)

        # Second conv layer post residual blocks.
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.coarse_dim, self.coarse_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.coarse_dim)
        )

        # 2 Upsampling layers.
        upsampling = []
        for _ in range(3):
            upsampling.append(UpsampleBlock(self.coarse_dim))

        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer.
        self.conv3 = nn.Conv2d(self.coarse_dim, self.nc, kernel_size=9, stride=1, padding=4)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(input)
        out = self.Trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        # out = self.sig(out)
        return out

    def sample_coarse(self, input: torch.Tensor) -> torch.Tensor:
        return input

class UpsampleBlock(nn.Module):
    r"""Main upsample block structure"""

    def __init__(self, channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.
        Args:
            channels (int): Number of channels in the input image.
        """
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out

class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.
        Args:
            channels (int): Number of channels in the input image.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)
        out = self.prelu(out)
        out = self.conv2(out)

        return out + input
