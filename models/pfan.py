
##############################################
# Imports
##############################################
import math
import os
import cv2
import logging

import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

from mmcv.cnn import build_norm_layer  # official SeaFormer / PFAN code依赖

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


##############################################
# 基础工具模块
##############################################
class LayerNorm(nn.Module):
    """LayerNorm 支持 channels_last 或 channels_first 两种格式."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def _make_divisible(v, divisor, min_value=None):
    """来自 MobileNet 官方实现的 channel 对齐函数."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic Depth / DropPath 实现."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


##############################################
# ECA 与 LeFF / ConvNeXt Block 系列
##############################################
class eca_layer_1d(nn.Module):
    """1D ECA 模块，用于 LeFF 中的通道注意力."""
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # x: [B, HW, C]
        y = self.avg_pool(x.transpose(-1, -2))           # [B, C, 1]
        y = self.conv(y.transpose(-1, -2))               # [B, 1, C]
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size
        return flops


class FastLeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, groups=hidden_dim),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)

        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = x.transpose(1, 2).view(bs, c, hh, hh)
        x = x.permute(0, 2, 3, 1)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.dim * self.hidden_dim
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.LeakyReLU,
                 drop=0., use_eca=True):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim,
                      kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)

        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)
        x = x.transpose(1, 2).view(bs, c, hh, hh)
        x = x.permute(0, 2, 3, 1)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.dim * self.hidden_dim
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


class BlockV2(nn.Module):
    """ConvNeXtV2 Block."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class Block(nn.Module):
    """ConvNeXt Block."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x = x1 + x2 + x3
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


##############################################
# Conv + MLP 基础模块 (SeaFormer 用)
##############################################
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


##############################################
# SeaFormer: Axial Positional Embedding + Sea_Attention
##############################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N,), mode='linear', align_corners=False)
        return x


class Sea_Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2, activation=nn.LeakyReLU,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = nn.Sequential(
            activation(),
            Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.proj_encode_row = nn.Sequential(
            activation(),
            Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = nn.Sequential(
            activation(),
            Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(
            2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
            groups=2 * self.dh, norm_cfg=norm_cfg
        )
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # row attention
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        # column attention
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        xx = xx.permute(0, 2, 3, 1)
        return xx


##############################################
# CBAM 注意力模块
##############################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1),
             torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16,
                 pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types
        )
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out   # ★ 修正：之前缺少 return，导致输出为 None


##############################################
# Swin / ViT 相关模块
##############################################
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(
            x, shifts=(self.displacement, self.displacement), dims=(1, 2)
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(
            mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2',
            h1=window_size, h2=window_size
        )
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(
        np.array([[x, y] for x in range(window_size) for y in range(window_size)])
    )
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim,
                 shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True, left_right=False
                ),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False, left_right=True
                ),
                requires_grad=False
            )

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(
                torch.randn(2 * window_size - 1, 2 * window_size - 1)
            )
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(window_size ** 2, window_size ** 2)
            )

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(
                t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                h=h, w_h=self.window_size, w_w=self.window_size
            ),
            qkv
        )

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(
            out,
            'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
            h=h, w_h=self.window_size, w_w=self.window_size,
            nw_h=nw_h, nw_w=nw_w
        )
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim,
                 shifted, window_size, relative_pos_embedding):
        super().__init__()
        # 官方 PFAN 使用 Sea_Attention 替代 Swin 的 self-attention
        self.attention_block = Residual(
            PreNorm(
                dim,
                Sea_Attention(dim=dim, num_heads=heads, key_dim=head_dim)
            )
        )
        self.mlp_block = Residual(
            PreNorm(dim, LeFF(dim=dim, hidden_dim=mlp_dim))
        )

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor,
            stride=downscaling_factor, padding=0
        )
        self.linear = nn.Linear(
            in_channels * downscaling_factor ** 2, out_channels
        )

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x)
        x = x.view(b, -1, new_h, new_w)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers,
                 downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2'

        self.patch_partition = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor
        )

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(
                    dim=hidden_dimension, heads=num_heads,
                    head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                    shifted=False, window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding
                ),
                SwinBlock(
                    dim=hidden_dimension, heads=num_heads,
                    head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                    shifted=True, window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding
                ),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class ViTs(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers,
                 downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        self.stage1 = StageModule(
            in_channels=in_channels,
            hidden_dimension=hidden_dimension,
            layers=layers,
            downscaling_factor=downscaling_factor,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dimension // 2, hidden_dimension, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )
        self.channel_att = ChannelGate(
            hidden_dimension, reduction_ratio=16, pool_types=['avg', 'max']
        )
        self.squeeze = nn.Sequential(
            nn.Conv2d(hidden_dimension, hidden_dimension // 2, 1, 1, 0),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        out = self.stage1(x)
        out = self.channel_att(out)
        # 官方实现中 fusion / squeeze 注释掉，这里保持一致
        return out + x


##############################################
# 轻量卷积块 (PFAN中使用)
##############################################
class ConvBlock(nn.Module):
    def __init__(self, inp, oup):
        super(ConvBlock, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(inp, oup, 3, 1, 1, groups=oup),
            nn.LeakyReLU(0.05)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(inp, oup, 5, 1, 2, groups=oup),
            nn.LeakyReLU(0.05)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(inp, oup, 7, 1, 3, groups=oup),
            nn.LeakyReLU(0.05)
        )

    def forward(self, x):
        return self.conv3(x) + self.conv5(x) + self.conv7(x)


class MulC(nn.Module):  # mobilenet style
    def __init__(self, inp, oup, exp=64, res=True):
        super(MulC, self).__init__()
        self.res = res
        conv_layer = nn.Conv2d
        nlin_layer = nn.LeakyReLU

        self.conv = nn.Sequential(
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            nlin_layer(inplace=True),
            ConvBlock(exp, exp),
            conv_layer(exp, oup, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        if self.res:
            return x + self.conv(x)
        else:
            return self.conv(x)


##############################################
# PFAN 主体网络
##############################################
def get_norm_layer(norm_type='batch'):
    """简单 norm 选择器，保持官方接口风格."""
    if isinstance(norm_type, str):
        if norm_type == 'batch':
            return nn.BatchNorm2d
        elif norm_type == 'instance':
            return nn.InstanceNorm2d
        else:
            raise NotImplementedError(f"Unknown norm type: {norm_type}")
    return norm_type


class PFAN(nn.Module):
    def __init__(self, *,
                 input_nc, output_nc, ngf,
                 hidden_dim, layers, heads,
                 channels=3, num_classes=1000, head_dim=32, window_size=8,
                 downscaling_factors=(1, 1, 1, 1),
                 relative_pos_embedding=True,
                 norm_layer_1='batch'):
        super().__init__()

        norm_layer_1 = get_norm_layer(norm_layer_1)

        model_1 = [
            nn.Conv2d(input_nc, ngf, 1, 1, 0),
            norm_layer_1(ngf),
            nn.LeakyReLU(0.05)
        ]

        model_3 = []
        model_3_1 = [nn.Tanh()]
        model_3 += [nn.Conv2d(hidden_dim, output_nc, 1, 1, 0)]

        self.model_1 = nn.Sequential(*model_1)

        self.convnext1 = Block(ngf)
        self.convnext2 = Block(ngf)

        # 注意：官方实现中 hidden_dim 通常等于 ngf
        self.vit = ViTs(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=4,
            relative_pos_embedding=relative_pos_embedding
        )

        self.model_3 = nn.Sequential(*model_3)
        self.model_3_1 = nn.Sequential(*model_3_1)

    def forward(self, img):
        x = self.model_1(img)

        x1 = self.convnext1(x)
        x1 = self.convnext2(x1)
        x2 = self.vit(x1) + x

        x3 = self.model_3(x2)
        x3 = self.model_3_1(x3)

        return x3


##############################################
# PFAN 变体构造函数
##############################################
def swin_t(hidden_dim=64, layers=(2, 2, 2, 2),
           heads=(4, 4, 4, 4), **kwargs):
    return PFAN(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2),
           heads=(3, 6, 12, 24), **kwargs):
    return PFAN(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2),
           heads=(4, 8, 16, 32), **kwargs):
    return PFAN(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2),
           heads=(6, 12, 24, 48), **kwargs):
    return PFAN(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)



