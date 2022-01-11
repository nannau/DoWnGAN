# Borrowed from:
# https://github.com/Lornatang/SRGAN-PyTorch/
# Adopted to ESRGAN: https://arxiv.org/abs/1809.00219

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchvision.models import vgg19
import math


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Generator(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(self, filters, fine_dims, channels, n_predictands=2, num_res_blocks=16, num_upsample=3):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, n_predictands, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


# class Generator(nn.Module):
#     r"""The main architecture of the generator."""

#     def __init__(self, coarse_dim, fine_dim, nc, n_predictands):
#         r"""This is an esrgan model defined by the author himself."""
#         super(Generator, self).__init__()
#         # First layer.
#         self.coarse_dim = coarse_dim
#         self.fine_dim = fine_dim
#         self.nc = nc
#         self.n_predictands = n_predictands
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(self.nc, self.coarse_dim, kernel_size=9, stride=1, padding=4),
#             nn.PReLU(),
#         )

#         # Residual blocks.
#         residual_blocks = []
#         for _ in range(16):
#             residual_blocks.append(ResidualBlock(self.coarse_dim))
#         self.Trunk = nn.Sequential(*residual_blocks)

#         # Second conv layer post residual blocks.
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 self.coarse_dim,
#                 self.coarse_dim,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(self.coarse_dim),
#         )

#         # 2 Upsampling layers.
#         upsampling = []
#         for _ in range(3):
#             upsampling.append(UpsampleBlock(self.coarse_dim))

#         self.upsampling = nn.Sequential(*upsampling)

#         # Final output layer.
#         self.conv3 = nn.Conv2d(
#             self.coarse_dim, self.n_predictands, kernel_size=9, stride=1, padding=4
#         )

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         out1 = self.conv1(input)
#         out = self.Trunk(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         out = self.upsampling(out)
#         out = self.conv3(out)
#         # out = self.sig(out)
#         return out

#     def sample_coarse(self, input: torch.Tensor) -> torch.Tensor:
#         return input


# class UpsampleBlock(nn.Module):
#     r"""Main upsample block structure"""

#     def __init__(self, channels):
#         r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.
#         Args:
#             channels (int): Number of channels in the input image.
#         """
#         super(UpsampleBlock, self).__init__()
#         self.conv = nn.Conv2d(
#             channels,
#             channels * 4,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False,
#         )
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
#         self.prelu = nn.PReLU()

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         out = self.conv(input)
#         out = self.pixel_shuffle(out)
#         out = self.prelu(out)

#         return out


# class ResidualBlock(nn.Module):
#     r"""Main residual block structure"""

#     def __init__(self, channels):
#         r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.
#         Args:
#             channels (int): Number of channels in the input image.
#         """
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             channels, channels, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(
#             channels, channels, kernel_size=3, stride=1, padding=1, bias=False
#         )

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         out = self.conv1(input)
#         out = self.prelu(out)
#         out = self.conv2(out)

#         return out + input
