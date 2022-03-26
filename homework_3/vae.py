from typing import Tuple

import torch
from torch import nn, Tensor


DOUBLE_TENSOR = Tuple[Tensor, Tensor]
QUATRO_TENSOR = Tuple[Tensor, Tensor, Tensor, Tensor]


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_transpose: bool = False):
        super().__init__()
        if is_transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class Vae64x64(nn.Module):
    def __init__(self, latent_dim: int, n_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            Block(n_channels, 32),  # -> [32; 32; 32]
            Block(32, 64),  # -> [16; 16; 64]
            Block(64, 128),  # -> [8; 8; 128]
            Block(128, 256),  # -> [4; 4; 256]
            Block(256, 512),  # -> [2; 2; 512]
            nn.Flatten(),  # -> [4 * 512]
            nn.Linear(4 * 512, 2 * latent_dim),  # -> [2 * latent dim]
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 512),  # -> [4 * 512]
            nn.Unflatten(1, (512, 2, 2)),  # -> [2; 2; 512]
            Block(512, 256, is_transpose=True),  # -> [4; 4; 256]
            Block(256, 128, is_transpose=True),  # -> [8; 8; 128]
            Block(128, 64, is_transpose=True),  # -> [16; 16; 64]
            Block(64, 32, is_transpose=True),  # -> [32; 32; 32]
            Block(32, 32, is_transpose=True),  # -> [64; 64; 32]
            nn.Conv2d(32, out_channels=n_channels, kernel_size=3, padding=1),  # -> [64; 64; 3]
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> DOUBLE_TENSOR:
        latent = self.encoder(x)
        mu, log_sigma = latent.chunk(2, dim=1)
        return mu, log_sigma

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(log_sigma) * torch.exp(0.5 * log_sigma)

    def forward(self, x: Tensor) -> QUATRO_TENSOR:
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        reconstruct = self.decode(z)
        return reconstruct, z, mu, log_sigma
