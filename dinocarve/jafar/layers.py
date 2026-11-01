"""Vendored from https://github.com/PaulCouairon/JAFAR @ 25321a2 (Apache-2.0, see LICENSE).
Changes: the four src/layers modules merged into this file; imports made package-relative."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(query_dim)
        self.norm_k = nn.RMSNorm(key_dim)
        self.norm_v = nn.RMSNorm(value_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            vdim=value_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, query, key, value):
        query = self.norm_q(query)
        key = self.norm_k(key)

        _, attn_scores = self.attention(query, key, self.norm_v(value), average_attn_weights=True)
        attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)

        return attn_output, attn_scores


class CrossAttentionBlock(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, num_heads, **kwargs):
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim,
            key_dim,
            value_dim,
            num_heads,
        )
        self.conv2d = nn.Conv2d(query_dim, query_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, **kwargs):
        q = self.conv2d(q)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")
        features, _ = self.cross_attn(q, k, v)

        return features


# Convolutions
class ResBlock(nn.Module):
    """Basic Residual Block, adapted from magvit1"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super(ResBlock, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=False,
            )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        return x + residual


class SFTModulation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.gamma = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.beta = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)

    def forward(self, image, features):
        gamma = self.gamma(features)
        beta = self.beta(features)
        return gamma * self.norm(image) + beta  # Spatial modulation


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()
