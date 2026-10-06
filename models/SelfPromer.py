import math
import logging
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


# =========================================================
# fema_utils: NormLayer / ActLayer / ResBlock / CombineQuantBlock
# =========================================================

class NormLayer(nn.Module):
    """Normalization layer: bn / in / gn / none"""

    def __init__(self, channels, norm_type='bn'):
        super().__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels

        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            raise ValueError(f'Norm type {norm_type} not supported.')

    def forward(self, x):
        return self.norm(x)


class ActLayer(nn.Module):
    """Activation layer: relu / leakyrelu / prelu / silu / gelu / none"""

    def __init__(self, channels, relu_type='leakyrelu'):
        super().__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            raise ValueError(f'Activation type {relu_type} not supported.')

    def forward(self, x):
        return self.func(x)


class ResBlock(nn.Module):
    """Pre-activation residual block（与原始 FeMaSR 相同设计）"""

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()
        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, x):
        res = self.conv(x)
        return res + x


class CombineQuantBlock(nn.Module):
    """
    用于多尺度 quant 特征 + 当前尺度 feature 融合。
    input1: 当前尺度 feature
    input2: 上一尺度 quant 特征（可为 None）
    """

    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_ch1 + in_ch2, out_channel, 3, 1, 1)

    def forward(self, input1, input2=None):
        if input2 is not None:
            input2 = F.interpolate(input2, input1.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((input1, input2), dim=1)
        else:
            x = input1
        return self.conv(x)


# =========================================================
# DCNv2 简化实现（不依赖自定义 C++/CUDA，基于 torchvision.ops.deform_conv2d）
# =========================================================

class ModulatedDeformConvPack(nn.Module):
    """
    简化版 DCNv2 基类：
    - conv_offset: 根据 feat 生成 offset + mask
    - forward 由子类实现
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 deformable_groups=1,
                 bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None

        # offset + mask
        self.conv_offset = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)
        nn.init.kaiming_uniform_(self.weight, a=1)

    def forward(self, x):
        raise NotImplementedError


class DCNv2Pack(ModulatedDeformConvPack):
    """
    与原 SelfPromer 一致接口：offset/mask 由 feat 生成，
    然后对 x 做 modulated deformable conv。
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(
                input=x,
                offset=offset,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask
            )
        else:
            raise RuntimeError("torchvision version < 0.9.0 不支持 deform_conv2d")


class DCNv2Pack_fusion(nn.Module):
    """
    SelfPromer 中的 DCN 融合模块：
    - 利用两个 DCNv2Pack，双向对齐 x 与 feat
    - 最后 concat 后 1x1 conv 融合
    """

    def __init__(self, in_channel1, in_channel2, out_channel, kernel=3, padding=1):
        super().__init__()
        self.fusion1 = nn.Conv2d(in_channel1 + in_channel2, out_channel, 1, 1)
        self.fusion2 = nn.Conv2d(in_channel1 + in_channel2, out_channel, 1, 1)

        self.dcn1 = DCNv2Pack(out_channel, out_channel, kernel_size=kernel, padding=padding)
        self.dcn2 = DCNv2Pack(out_channel, out_channel, kernel_size=kernel, padding=padding)

        self.fusion = nn.Conv2d(out_channel * 2, out_channel, 1, 1)

    def forward(self, x, feat):
        f1 = self.fusion1(torch.cat([x, feat], dim=1))
        o1 = self.dcn1(x, f1)

        f2 = self.fusion2(torch.cat([feat, x], dim=1))
        o2 = self.dcn2(f2, x)

        return self.fusion(torch.cat([o1, o2], dim=1))


# =========================================================
# VQ 模块（与原 FeMaSRNet 一致）
# =========================================================

class VectorQuantizer(nn.Module):
    """
    与 SelfPromer / FeMaSR 中一致的 VQ 模块。
    支持：
      - 普通 commitment loss
      - LQ_stage 时基于 GT code index 的 codebook + Gram loss
    """

    def __init__(self, n_e, e_dim, beta=0.25, LQ_stage=False):
        super().__init__()
        self.n_e = int(n_e)
        self.e_dim = int(e_dim)
        self.LQ_stage = LQ_stage
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    @staticmethod
    def dist(x, y):
        # x: (N, C)  y: (K, C)
        return (
                torch.sum(x ** 2, dim=1, keepdim=True) +
                torch.sum(y ** 2, dim=1) -
                2 * torch.matmul(x, y.t())
        )

    @staticmethod
    def gram_loss(x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)
        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)
        return (gmx - gmy).pow(2).mean()

    def forward(self, z, gt_indices=None, current_iter=None):
        # z: (B, C, H, W)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.e_dim)

        codebook = self.embedding.weight  # (K, C)
        d = self.dist(z_flat, codebook)

        # 最近 codebook index
        min_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_onehot = torch.zeros(min_indices.shape[0], codebook.shape[0], device=z.device)
        min_onehot.scatter_(1, min_indices, 1)

        # 若有 GT indices，构造对应 onehot
        if gt_indices is not None:
            gt_indices = gt_indices.reshape(-1)
            gt_min_indices = gt_indices.reshape_as(min_indices)
            gt_onehot = torch.zeros_like(min_onehot)
            gt_onehot.scatter_(1, gt_min_indices, 1)

            z_q_gt = torch.matmul(gt_onehot, codebook).view_as(z)

        # 量化
        z_q = torch.matmul(min_onehot, codebook).view_as(z)

        # loss
        e_latent_loss = (z_q.detach() - z).pow(2).mean()
        q_latent_loss = (z_q - z.detach()).pow(2).mean()

        if self.LQ_stage and gt_indices is not None:
            codebook_loss = self.beta * (z_q_gt.detach() - z).pow(2).mean()
            texture_loss = self.gram_loss(z, z_q_gt.detach())
            codebook_loss = codebook_loss + texture_loss
        else:
            codebook_loss = q_latent_loss + self.beta * e_latent_loss

        # straight-through
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, codebook_loss, min_indices.reshape(z_q.shape[0], 1, z_q.shape[2], z_q.shape[3])

    def get_codebook_entry(self, indices):
        """
        indices: (B,1,H,W)
        """
        b, _, h, w = indices.shape
        indices = indices.view(-1).to(self.embedding.weight.device)

        onehot = torch.zeros(indices.shape[0], self.n_e, device=indices.device)
        onehot.scatter_(1, indices[:, None], 1)

        z_q = torch.matmul(onehot, self.embedding.weight)
        z_q = z_q.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return z_q


# =========================================================
# Multi-scale Encoder / DecoderBlock / Transformer Blocks
# =========================================================

class MultiScaleEncoder(nn.Module):
    """
    与原 FeMaSR 一致的多尺度 Encoder：
    - LQ_stage=False: 单输入
    - LQ_stage=True: 可以 depth 引导（input_cat 模式）
    """

    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 depth_guide='input_cat',
                 **kwargs):
        super().__init__()
        self.LQ_stage = LQ_stage
        self.depth_guide = depth_guide
        ksz = 3

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.max_depth = max_depth

        if not LQ_stage:
            self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)
            res = input_res
            for _ in range(max_depth):
                in_ch = channel_query_dict[res]
                out_ch = channel_query_dict[res // 2]
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                        ResBlock(out_ch, out_ch, norm_type, act_type),
                        ResBlock(out_ch, out_ch, norm_type, act_type),
                    )
                )
                res //= 2
        else:
            if self.depth_guide == 'input_cat':
                self.in_conv = nn.Sequential(
                    nn.Conv2d(in_channel * 2, channel_query_dict[input_res], 3, padding=1),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                )
            else:
                self.in_conv = nn.Sequential(
                    nn.Conv2d(in_channel, channel_query_dict[input_res], 3, padding=1),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                    ResBlock(channel_query_dict[input_res], channel_query_dict[input_res], norm_type, act_type),
                )

            res = input_res
            for _ in range(max_depth):
                in_ch = channel_query_dict[res]
                out_ch = channel_query_dict[res // 2]
                self.blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, ksz, stride=2, padding=1),
                        ResBlock(out_ch, out_ch, norm_type, act_type),
                        ResBlock(out_ch, out_ch, norm_type, act_type),
                    )
                )
                res //= 2

    def forward(self, x, depth=None):
        feats = []
        if self.LQ_stage:
            if depth is not None and self.depth_guide == 'input_cat':
                x = self.in_conv(torch.cat([x, depth], dim=1))
            else:
                x = self.in_conv(x)
        else:
            b, c, h, w = x.size()
            x = self.in_conv(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        for blk in self.blocks:
            x = blk(x)
            feats.append(x)

        # 与原代码一致：返回从小到大的尺度
        return feats


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        )

    def forward(self, x):
        return self.block(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, prompt=None, H=None, W=None):
        B, N, C = x.shape
        if prompt is not None:
            q = x + prompt
        else:
            q = x
        q = self.q(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        from timm.models.layers import DropPath  # timm 依赖

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, prompt=None, H=None, W=None):
        x = x + self.drop_path(self.attn(self.norm1(x), prompt, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =========================================================
# FeMaSRNet / SelfPromer 主结构（端到端）
# =========================================================

class FeMaSRNet(nn.Module):
    """
    按原 SelfPromer / FeMaSRNet 结构改写，去掉 basicsr 依赖。
      - Multi-scale encoder
      - Depth prompt (LQ_stage=True 时启用) + Transformer
      - 多尺度 VQ codebook
      - DCN 融合 + U-Net style decoder（仅 LQ_stage=True 时启用 DCN + upsampler）
    """

    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 scale_factor=1,
                 use_residual=True,
                 position='depth',
                 prompt=True,
                 n_layers=16,
                 depth_guide='prompt',
                 **ignore_kwargs):
        super().__init__()

        if codebook_params is None:
            # 默认单尺度 codebook 配置：32×32 -> 1024 codebook, dim=256
            codebook_params = [[32, 1024, 256]]

        codebook_params = np.array(codebook_params)
        self.codebook_scale = codebook_params[:, 0]
        codebook_emb_num = codebook_params[:, 1].astype(int)
        codebook_emb_dim = codebook_params[:, 2].astype(int)

        self.prompt = prompt
        self.position = position
        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = scale_factor if LQ_stage else 1
        self.use_residual = use_residual
        self.depth_guide = depth_guide

        # 通道表，尽量与原版本一致
        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }
        self.channel_query_dict = channel_query_dict  # 若后面想调试也方便看

        # DCN fusion：三层，与原代码相同（仅 LQ_stage=True 时在 forward 中实际使用）
        self.dcn_fusion = nn.ModuleList()
        self.dcn_fusion.append(
            DCNv2Pack_fusion(
                in_channel1=channel_query_dict[32],
                in_channel2=channel_query_dict[32],
                out_channel=channel_query_dict[32]
            )
        )
        self.dcn_fusion.append(
            DCNv2Pack_fusion(
                in_channel1=channel_query_dict[32],
                in_channel2=channel_query_dict[64],
                out_channel=channel_query_dict[64]
            )
        )
        self.dcn_fusion.append(
            DCNv2Pack_fusion(
                in_channel1=channel_query_dict[32],
                in_channel2=channel_query_dict[128],
                out_channel=channel_query_dict[128]
            )
        )

        # LQ_stage 下的 upsampler（texture 分支，HQ 不会用到）
        if LQ_stage:
            res = 32
            self.upsampler = nn.ModuleList()
            out_ch_tex = channel_query_dict[res]
            self.upsampler.append(
                nn.Sequential(
                    ResBlock(out_ch_tex, out_ch_tex, norm_type, act_type),
                    ResBlock(out_ch_tex, out_ch_tex, norm_type, act_type),
                )
            )
            res = 64
            for _ in range(2):
                out_ch_tex = channel_query_dict[res]
                self.upsampler.append(
                    nn.Sequential(
                        ResBlock(out_ch_tex, out_ch_tex, norm_type, act_type),
                        ResBlock(out_ch_tex, out_ch_tex, norm_type, act_type),
                    )
                )
                res *= 2

        # 多尺度编码深度
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale[0]))
        encode_depth = int(np.log2(gt_resolution // self.scale_factor // self.codebook_scale[0]))

        # Encoder
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            encode_depth,
            self.gt_res // self.scale_factor,
            channel_query_dict,
            norm_type,
            act_type,
            LQ_stage,
            depth_guide=self.depth_guide
        )

        # Decoder
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // (2 ** self.max_depth) * (2 ** i)
            in_ch = channel_query_dict[res]
            out_ch = channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))
        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # Transformer prompt 部分
        self.dim_embd = channel_query_dict[32]
        self.n_head = 8
        self.n_layers = n_layers
        self.transformer = nn.Sequential(
            *[Block(dim=self.dim_embd, num_heads=self.n_head) for _ in range(self.n_layers)]
        )

        # 多尺度 VQ 组
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(codebook_params.shape[0]):
            quant = VectorQuantizer(
                codebook_emb_num[scale],
                codebook_emb_dim[scale],
                LQ_stage=self.LQ_stage,
            )
            self.quantize_group.append(quant)

            scale_in_ch = channel_query_dict[self.codebook_scale[scale]]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_in_ch1 = codebook_emb_dim[scale]
                comb_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_in_ch1 = codebook_emb_dim[scale - 1]
                comb_in_ch2 = codebook_emb_dim[scale]

            self.before_quant_group.append(
                nn.Conv2d(quant_conv_in_ch, codebook_emb_dim[scale], 1)
            )
            self.after_quant_group.append(
                CombineQuantBlock(comb_in_ch1, comb_in_ch2, scale_in_ch)
            )

    # =========================================================
    # =============== HQ-stage（无 depth）逻辑 =================
    # =========================================================
    def encode_and_decode_HQ(self, input):
        """
        HQ-stage：高质量阶段，不使用 depth / DCN / upsampler / depth prompt。
        用于你现在的对比实验（单输入图像 restoration）。
        """
        # 1) Multi-scale encoder
        enc_feats = self.multiscale_encoder(input)   # 返回从大到小 or 小到大取决于 MultiScaleEncoder，这里我们保持与前面一致使用原始输出顺序
        enc_feats = enc_feats[::-1]                  # 小 → 大顺序

        # 2) Transformer 输入为最小尺度特征（例如 32×32）
        x = enc_feats[0]
        B, C, H, W = x.shape

        xf = x.flatten(2).transpose(1, 2)           # (B, N, C)
        prompt = None                               # HQ-stage 不使用 prompt

        for blk in self.transformer:
            xf = blk(xf, prompt, H, W)

        x = xf.transpose(1, 2).reshape(B, C, H, W)

        # 3) VQ（默认只用第 0 个 codebook）
        z_pre = self.before_quant_group[0](x)
        z_q, codebook_loss, indices = self.quantize_group[0](z_pre)
        x = self.after_quant_group[0](z_q)

        # 4) Decoder（HQ-stage 不做 DCN fusion / upsampler）
        for dec in self.decoder_group:
            x = dec(x)

        out = self.out_conv(x)

        # 为了与 LQ-stage 的 encode_and_decode 返回结构对齐，后面几个量直接返回 0 / None
        zero = torch.tensor(0.0, device=out.device)
        return (
            out,              # out_img
            zero,             # codebook_loss（在 HQ 可以不关心这个，只用 codebook_loss_second）
            codebook_loss,    # codebook_loss_second：这里用真正的 VQ loss 放在第二个方便你记录
            None,             # indices
            indices,          # indices_second
            None, None,       # depth_quant_1, depth_quant_2
            None, None        # depth1, depth2
        )

    # =========================================================
    # =============== LQ-stage（原 depth + DCN 逻辑） ==========
    # =========================================================
    def encode_and_decode_LQ(self,
                             input,
                             gt_img=None,
                             depth1=None,
                             depth2=None,
                             depth_quant_1=None,
                             depth_quant_2=None,
                             gt_indices=None,
                             current_iter=None):
        """
        完整保留原 LQ-stage 逻辑：depth prompt + DCN fusion + upsampler。
        若你只做 HQ 对比，可以完全不用管这个分支。
        """
        codebook_loss_list = []
        indices = None
        codebook_loss_list_second = []
        indices_second = None

        # Multi-scale 特征
        if self.LQ_stage and self.depth_guide == 'input_cat' and depth1 is not None:
            depth1_cat = torch.cat([depth1, depth1, depth1], dim=1)
            depth1_cat = torch.clamp(depth1_cat, 0, 1)
            enc_feats1 = self.multiscale_encoder(input.detach(), depth=depth1_cat)
        else:
            enc_feats1 = self.multiscale_encoder(input.detach())
        enc_feats1 = enc_feats1[::-1]  # 小->大顺序

        x1 = enc_feats1[0]
        z_quant = x1  # base feature

        # 如果没有传入 depth_quant_1/2，这里 LQ-stage 默认退化为“无 prompt”（防止 None 报错）
        if (depth_quant_1 is None) or (depth_quant_2 is None):
            prompt = torch.zeros_like(z_quant)
        else:
            depth_error = torch.abs(depth_quant_1 - depth_quant_2)
            prompt = depth_error * z_quant

        x = z_quant
        b, c, h, w = x.size()
        x_flat = x.flatten(2).transpose(1, 2)       # (B,N,C)
        prompt_flat = prompt.flatten(2).transpose(1, 2)

        # transformer
        x_flat = x_flat + prompt_flat
        for blk in self.transformer:
            x_flat = blk(x_flat, prompt_flat, h, w)
        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        trans_feature = x  # 作为上采样 texture

        # 量化（这里只实现 scale=0）
        z_quant0 = self.before_quant_group[0](x)
        z_quant2, codebook_loss2, indices2 = self.quantize_group[0](z_quant0, gt_indices)
        indices_second = indices2
        codebook_loss_list_second.append(codebook_loss2)
        z_quant = z_quant2

        # combine quant + spatial feature
        x = self.after_quant_group[0](z_quant)

        # Decoder + DCN fusion + upsampler
        for i in range(self.max_depth):
            if self.LQ_stage and self.use_residual and i < len(self.dcn_fusion):
                up = F.interpolate(trans_feature, size=enc_feats1[i].shape[2:], mode='bilinear',
                                   align_corners=False)
                dcn = self.dcn_fusion[i](enc_feats1[i], up)
                up = self.upsampler[i](dcn)
                x = x + up
            x = self.decoder_group[i](x)

        out_img = self.out_conv(x)

        codebook_loss = sum(codebook_loss_list) if len(codebook_loss_list) else torch.tensor(0.0, device=out_img.device)
        codebook_loss_second = sum(codebook_loss_list_second)

        return (
            out_img,
            codebook_loss,
            codebook_loss_second,
            indices,
            indices_second,
            depth_quant_1,
            depth_quant_2,
            depth1,
            depth2
        )

    # =========================================================
    # =============== 统一对外 encode_and_decode 接口 ==========
    # =========================================================
    def encode_and_decode(self,
                          input,
                          gt_img=None,
                          depth1=None,
                          depth2=None,
                          depth_quant_1=None,
                          depth_quant_2=None,
                          gt_indices=None,
                          current_iter=None):
        """
        统一入口：
          - 若 self.LQ_stage=False → 走 HQ-stage（用于你现在的对比实验）
          - 若 self.LQ_stage=True  → 走原 LQ-stage（需要 depth / DCN 的那套）
        """
        if not self.LQ_stage:
            return self.encode_and_decode_HQ(input)
        else:
            return self.encode_and_decode_LQ(
                input,
                gt_img=gt_img,
                depth1=depth1,
                depth2=depth2,
                depth_quant_1=depth_quant_1,
                depth_quant_2=depth_quant_2,
                gt_indices=gt_indices,
                current_iter=current_iter
            )

    # =========================================================
    # =============== 其余外部接口保持不变 =====================
    # =========================================================
    def decode_indices(self, indices):
        """从 code index 直接 decode 一张图（用于可视化 codebook）"""
        assert len(indices.shape) == 4
        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)
        for dec in self.decoder_group:
            x = dec(x)
        return self.out_conv(x)

    def check_image_size(self, x, padding=8):
        _, _, h, w = x.size()
        mod_pad_h = (padding - h % padding) % padding
        mod_pad_w = (padding - w % padding) % padding
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    @torch.no_grad()
    def test(self,
             input,
             second_img=None,
             depth1=None,
             depth2=None,
             depth_quant_1=None,
             depth_quant_2=None):
        b, c, h_old, w_old = input.size()
        input = self.check_image_size(input, padding=8)

        dec, _, _, _, _, _, _, _, _ = self.encode_and_decode(
            input=input,
            gt_img=second_img,
            depth1=depth1,
            depth2=depth2,
            depth_quant_1=depth_quant_1,
            depth_quant_2=depth_quant_2
        )

        dec = dec[..., :h_old, :w_old]
        return dec

    def forward(self,
                input,
                second_img=None,
                depth1=None,
                depth2=None,
                depth_quant_1=None,
                depth_quant_2=None,
                gt_indices=None):
        """
        返回单一张量 out_img，以适配你的训练主程序。
        所有额外信息（codebook_loss、indices 等）忽略。
        """
        out_img, _, _, _, _, _, _, _, _ = self.encode_and_decode(
            input,
            gt_img=second_img,
            depth1=depth1,
            depth2=depth2,
            depth_quant_1=depth_quant_1,
            depth_quant_2=depth_quant_2,
            gt_indices=gt_indices
        )
        return out_img


class SelfPromerWrapper(nn.Module):
    """
    统一 pipeline 使用的 SelfPromer 版本：
      - 内部调用官方 FeMaSRNet（LQ_stage=True，保留 DCN + VQ + Transformer 结构）
      - 不使用真实 depth / gt_indices，只构造“零 prompt”
      - forward(inp) 只返回一张复原图 pred，方便与其他模型公平对比
    """

    def __init__(self,
                 in_channel=3,
                 codebook_params=None,
                 gt_resolution=256,
                 prompt=True,
                 n_layers=16,
                 norm_type='gn',
                 act_type='silu'):
        super().__init__()

        if codebook_params is None:
            codebook_params = [[32, 1024, 256]]

        # 使用官方 FeMaSRNet，开启 LQ_stage，但 depth_guide='prompt'，不拼 depth
        self.backbone = FeMaSRNet(
            in_channel=in_channel,
            codebook_params=codebook_params,
            gt_resolution=gt_resolution,
            LQ_stage=True,          # ★ 使用 LQ 分支（包含 DCN + upsampler）
            norm_type=norm_type,
            act_type=act_type,
            use_quantize=True,
            scale_factor=1,
            use_residual=True,
            position='depth',
            prompt=prompt,
            n_layers=n_layers,
            depth_guide='prompt',   # ★ 不 input_cat depth
        )

        # 方便计算深度特征形状：SelfPromer 里 dim_embd 对应 32×32 那一层的通道数
        self.code_scale = int(self.backbone.codebook_scale[0])  # 通常是 32
        self.embed_dim = int(self.backbone.dim_embd)            # 通常是 256
        self.gt_res = int(gt_resolution)

    def forward(self, x):
        """
        x: (B, 3, H, W)，在你的 pipeline 里 H=W=PS_W=gt_resolution
        返回：pred 图像 (B, 3, H, W)
        """
        B, C, H, W = x.shape

        # -------- 构造“零 depth_quant_1/2”，只用来提供形状，prompt=0 --------
        # SelfPromer 的 codebook_scale[0] 对应的空间尺寸就是 32（官方设计）
        S = self.code_scale        # e.g. 32
        Cq = self.embed_dim        # e.g. 256

        # 这里只需要在计算图外构造即可，不参与梯度
        depth_quant_1 = x.new_zeros((B, Cq, S, S))
        depth_quant_2 = x.new_zeros((B, Cq, S, S))

        # -------- 调用官方 encode_and_decode，保留结构，但不使用 GT / depth --------
        dec, _, _, _, _, _, _, _, _ = self.backbone.encode_and_decode(
            input=x,
            gt_img=None,
            depth1=None,
            depth2=None,
            depth_quant_1=depth_quant_1,
            depth_quant_2=depth_quant_2,
            gt_indices=None,
            current_iter=None,
        )

        return dec