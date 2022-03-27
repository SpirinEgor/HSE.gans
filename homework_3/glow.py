import math
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

DOUBLE_TENSOR = Tuple[Tensor, Tensor]
QUATRO_TENSOR = Tuple[Tensor, Tensor, Tensor, Tensor]

GLOW_OUT = Tuple[Tensor, Tensor, List[Tensor]]


def squeeze(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * 4, h // 2, w // 2)
    return x


def unsqueeze(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, c // 4, h * 2, w * 2)
    return x


def gaussian_log_p(x: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    return -0.5 * math.log(2 * math.pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


def gaussian_sample(eps: Tensor, mean: Tensor, log_std: Tensor):
    return mean + torch.exp(log_std) * eps


class ActNorm(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        self.log_s = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self._initialized = False

    def _init_weights(self, x: Tensor):
        with torch.no_grad():
            s = torch.std(x, dim=0, keepdim=True)
            self.log_s.data = -torch.log(s)
            self.b.data = -torch.mean(x, dim=0, keepdim=True) * self.log_s.exp()
            self._initialized = True

    def forward(self, x: Tensor) -> DOUBLE_TENSOR:
        """
        log det: h * w * sum(log|s|)
        """
        if not self._initialized:
            self._init_weights(x)

        out = x * self.log_s.exp() + self.b

        _, _, height, width = x.shape
        log_det = torch.sum(self.log_s)

        return out, log_det

    @torch.no_grad()
    def reverse(self, y: torch.Tensor) -> Tensor:
        out = (y - self.b) * torch.exp(-self.log_s)
        return out


class InvariantConv2d(nn.Module):
    """This cost of computing det(W) can be reduced from O(c^3) to O(c)
    by parameterizing W directly in its LU decomposition
        W = PL(U + diag(s))
    """

    def __init__(self, n_channels: int):
        """In this parameterization, we initialize the parameters by
        1. Sampling a random rotation matrix W
        2. Computing the corresponding value of P (which remains fixed)
        3. Computing the corresponding initial values of L and U and s (which are optimized).
        """
        super().__init__()
        self.dim = n_channels

        Q = nn.init.orthogonal_(torch.rand((n_channels, n_channels)))
        P, L, U = torch.lu_unpack(*Q.lu())

        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

    def calc_weights(self, device: torch.device) -> Tensor:
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.dim, device=device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S).to(device))
        return W.view(self.dim, self.dim, 1, 1)

    def forward(self, x: torch.Tensor) -> DOUBLE_TENSOR:
        """
        log det: h * w * sum(log(|s|))
        """
        weights = self.calc_weights(x.device)
        out = F.conv2d(x, weights)

        _, _, height, width = x.shape
        log_det = height * width * torch.sum(torch.log(torch.abs(self.S)))

        return out, log_det

    @torch.no_grad()
    def reverse(self, y: torch.Tensor) -> Tensor:
        weights = self.calc_weights(y.device).squeeze()
        weights = weights.inverse()
        weights = weights.view(self.dim, self.dim, 1, 1)

        return F.conv2d(y, weights)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(x)
        return out * torch.exp(self.scale * 3)


class NormalizedConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.conv(batch)


class AffineCoupling(nn.Module):
    def __init__(self, in_channels: int, n_filters: int):
        """
        We initialize the last convolution of each NN() with zeros,
        such that each affine coupling layer initially performs an identity function;
        we found that this helps training very deep networks.
        """
        super().__init__()

        self.net = nn.Sequential(
            NormalizedConv2D(in_channels // 2, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            NormalizedConv2D(n_filters, n_filters, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(n_filters, in_channels),
        )

    def forward(self, x: Tensor) -> DOUBLE_TENSOR:
        """
        log det: sum(log(|s|))
        """
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.net(x_a).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        y_b = (x_b + t) * s
        out = torch.cat([x_a, y_b], 1)

        log_det = torch.sum(torch.log(s), dim=[1, 2, 3])
        return out, log_det

    @torch.no_grad()
    def reverse(self, y: Tensor) -> Tensor:
        y_a, y_b = y.chunk(2, 1)
        log_s, t = self.net(y_a).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        x_b = y_b / s - t

        return torch.cat([y_a, x_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channels: int, n_filters: int):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.inv_conv2d = InvariantConv2d(in_channels)
        self.coupling = AffineCoupling(in_channels, n_filters)

    def forward(self, x: Tensor) -> DOUBLE_TENSOR:
        log_det = torch.zeros(x.shape[0], device=x.device)
        for block in [self.actnorm, self.inv_conv2d, self.coupling]:
            x, cur_log_det = block(x)
            log_det += cur_log_det
        return x, log_det

    @torch.no_grad()
    def reverse(self, y: Tensor) -> Tensor:
        for block in [self.coupling, self.inv_conv2d, self.actnorm]:
            y = block.reverse(y)
        return y


class GlowBlock(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, k_flow: int, is_last: bool = False):
        super().__init__()
        self.flow_blocks = nn.ModuleList()
        for _ in range(k_flow):
            self.flow_blocks.append(Flow(in_channels * 4, n_filters))

        self._is_last = is_last
        self.prior = (
            ZeroConv2d(in_channels * 4, in_channels * 8) if is_last else ZeroConv2d(in_channels * 2, in_channels * 4)
        )

    def forward(self, x: Tensor) -> QUATRO_TENSOR:
        bs = x.shape[0]

        # 1. Squeeze
        x = squeeze(x)

        # 2. Flow
        log_det = 0
        for flow in self.flow_blocks:
            x, cur_log_det = flow(x)
            log_det += cur_log_det

        # 3. Split (only if not last block)
        if self._is_last:
            zero = torch.zeros_like(x)
            mean, log_std = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(x, mean, log_std)
            log_p = log_p.view(bs, -1).sum(dim=1)
            out, z_new = x, x
        else:
            out, z_new = x.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_std)
            log_p = log_p.view(bs, -1).sum(dim=1)

        return out, z_new, log_p, log_det

    @torch.no_grad()
    def reverse(self, y: Tensor, eps: Tensor, is_reconstruct: bool = False) -> Tensor:
        if is_reconstruct:
            if not self._is_last:
                y = torch.cat([y, eps], dim=1)
        else:
            if self._is_last:
                zero = torch.zeros_like(y)
                mean, log_std = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                y = z
            else:
                mean, log_std = self.prior(y).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_std)
                y = torch.cat([y, z], dim=1)

        for flow in reversed(self.flow_blocks):
            y = flow.reverse(y)

        return unsqueeze(y)


class Glow(nn.Module):
    def __init__(self, k_flow: int, l_block: int, n_filters: int, n_channels: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        cur_channels = n_channels
        for _ in range(l_block - 1):
            self.blocks.append(GlowBlock(cur_channels, n_filters, k_flow))
            cur_channels *= 2
        self.blocks.append(GlowBlock(cur_channels, n_filters, k_flow, is_last=True))

    def forward(self, x: Tensor) -> GLOW_OUT:
        z_outs = []
        log_det = 0
        log_p = 0

        for block in self.blocks:
            x, z_new, log_p_new, log_det_new = block(x)
            z_outs.append(z_new)
            log_det += log_det_new
            log_p += log_p_new

        return log_p, log_det, z_outs

    @torch.no_grad()
    def reverse(self, z_list: List[Tensor], is_reconstruct: bool = False) -> Tensor:
        x = None
        for z, block in zip(z_list[::-1], self.blocks[::-1]):
            if x is None:
                x = block.reverse(z, z, is_reconstruct)
            else:
                x = block.reverse(x, z, is_reconstruct)
        return x

    def get_sample_noise_shapes(self, img_size: int, n_channels: int = 3) -> List:
        z_shapes = []
        for _ in range(len(self.blocks) - 1):
            n_channels *= 2
            img_size //= 2
            z_shapes.append((n_channels, img_size, img_size))
        z_shapes.append((n_channels * 4, img_size // 2, img_size // 2))
        return z_shapes

    def sample(self, z_noise: List[Tensor]) -> Tensor:
        with torch.no_grad():
            sampled_images = self.reverse(z_noise)
            sampled_images = sampled_images.clamp(-0.5, 0.5) + 0.5
        return sampled_images
