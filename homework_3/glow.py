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

        self.s = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self._initialized = False

    def _init_weights(self, x: Tensor):
        with torch.no_grad():
            # [1; n channels; 1; 1]
            s = torch.std(x, dim=(0, 2, 3), keepdim=True)
            self.s.data = 1 / (s + 1e-6)
            self.b.data = -torch.mean(x, dim=(0, 2, 3), keepdim=True)
            self._initialized = True

    def forward(self, x: Tensor) -> DOUBLE_TENSOR:
        """
        log det: h * w * sum(log|s|)
        """
        if not self._initialized:
            self._init_weights(x)

        out = (x + self.b) * self.s

        _, _, height, width = x.shape
        log_abs_s = torch.log(torch.abs(self.s))
        log_det = height * width * torch.sum(log_abs_s)

        return out, log_det

    @torch.no_grad()
    def reverse(self, y: torch.Tensor) -> Tensor:
        out = y / self.s - self.b
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
        shape = (n_channels, n_channels)

        w, _ = torch.linalg.qr(torch.rand(shape))

        w_p, lower, upper = torch.lu_unpack(*torch.lu(w))
        self.register_buffer("w_p", w_p)

        self.upper = nn.Parameter(upper)
        self.register_buffer("u_mask", torch.triu(torch.ones(shape), 1))

        self.lower = nn.Parameter(lower)
        self.register_buffer("l_mask", torch.tril(torch.ones(shape), -1))
        self.register_buffer("eye", torch.eye(*shape))

        s = torch.diag(upper)
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def calc_weights(self) -> Tensor:
        lower = self.lower * self.l_mask + self.eye
        upper = self.upper * self.u_mask
        s = torch.diag(self.sign_s * self.log_s.exp())

        weight = self.w_p @ lower @ (upper + s)
        return weight.view(*weight.shape, 1, 1)

    def forward(self, x: torch.Tensor) -> DOUBLE_TENSOR:
        """
        log det: h * w * sum(log(|s|))
        """
        weights = self.calc_weights()
        out = F.conv2d(x, weights)

        _, _, height, width = x.shape
        log_det = height * width * torch.sum(self.log_s)

        return out, log_det

    @torch.no_grad()
    def reverse(self, y: torch.Tensor) -> Tensor:
        weights = self.calc_weights().squeeze()
        weights = weights.inverse()
        weights = weights.view(*weights.shape, 1, 1)

        return F.conv2d(y, weights)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return out * torch.exp(self.scale * 3)


class AffineCoupling(nn.Module):
    def __init__(self, in_channels: int, n_filters: int):
        """
        We initialize the last convolution of each NN() with zeros,
        such that each affine coupling layer initially performs an identity function;
        we found that this helps training very deep networks.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, 1),
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

        log_det = torch.sum(torch.log(s).view(x.shape[0], -1), dim=1)
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
        log_det = None
        for block in [self.actnorm, self.inv_conv2d, self.coupling]:
            x, cur_log_det = block(x)
            log_det = cur_log_det if log_det is None else (log_det + cur_log_det)
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
            out, z_new = x, x
        else:
            out, z_new = x.chunk(2, 1)
            mean, log_std = self.prior(out).chunk(2, 1)

        log_p = gaussian_log_p(z_new, mean, log_std)
        log_p = log_p.sum(dim=[1, 2, 3])

        return out, z_new, log_p, log_det

    @torch.no_grad()
    def reverse(self, y: Tensor, eps: Tensor, is_reconstruct: bool = False) -> Tensor:
        if is_reconstruct:
            y = eps if self._is_last else torch.cat([y, eps], dim=1)
        else:
            if self._is_last:
                mean, log_std = self.prior(torch.zeros_like(y)).chunk(2, 1)
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
    def __init__(self, n_channels: int, n_filters: int, k_flow: int, l_block: int, is_linear: bool = False):
        super().__init__()
        self._is_linear = is_linear

        self.blocks = nn.ModuleList()
        cur_channels = n_channels
        for _ in range(l_block - 1):
            self.blocks.append(GlowBlock(cur_channels, n_filters, k_flow))
            cur_channels *= 2
        self.blocks.append(GlowBlock(cur_channels, n_filters, k_flow, is_last=True))

    def forward(self, x: Tensor) -> GLOW_OUT:
        if self._is_linear:
            bs, n = x.shape
            # [bs; 1; n; n]
            x = x.unsqueeze(1).expand(bs, n, n).unsqueeze(1)

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

        if self._is_linear:
            x = x[:, 0, 0]
        return x

    def sample(self, n_samples: int, n_channels: int, img_size: int, device: torch.device) -> Tensor:
        z_shapes = []
        for _ in range(len(self.blocks) - 1):
            n_channels *= 2
            img_size //= 2
            z_shapes.append((n_channels, img_size, img_size))
        z_shapes.append((n_channels * 4, img_size // 2, img_size // 2))

        z_samples = []
        for shape in z_shapes:
            z_samples.append(torch.randn(n_samples, *shape).to(device))

        return self.reverse(z_samples)
