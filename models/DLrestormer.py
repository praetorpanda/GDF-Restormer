import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

##############################################
# LayerNorm
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


##############################################
# Dilated Depthwise Convolution Module
class DilatedDWConv(nn.Module):
    def __init__(self, channels, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)


##############################################
# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, bias, dilation=1):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = DilatedDWConv(hidden * 2, dilation=dilation, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


##############################################
# Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h1=H, w1=W)
        return self.project_out(out)


##############################################
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expansion_factor, bias, LayerNorm_type, dilation=1):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, expansion_factor, bias, dilation)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##############################################
# Patch Embedding
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


##############################################
# Downsample & Upsample
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)
    

class ConvBlock(nn.Module):
    def __init__(self, dim, dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x):
        return x + self.block
    



import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossConvBlock(nn.Module):
    def __init__(self, dim, dilation=2, kernel_size=7, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.branch_dilated = nn.Conv2d(dim, dim, 3, 1, padding=dilation, dilation=dilation, bias=bias)

        # padding=0，手动填充
        self.branch_row_conv = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=0, bias=bias)
        self.branch_col_conv = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=0, bias=bias)

        self.norm = nn.GroupNorm(1, dim)
        self.relu = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.res = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        d = self.branch_dilated(x)

        pad_w = self.kernel_size // 2

        # 左右对称 pad：padding=(left, right, top, bottom)
        r = F.pad(x, (pad_w, pad_w, 0, 0), mode='reflect')
        r = self.branch_row_conv(r)

        # 上下对称 pad
        c = F.pad(x, (0, 0, pad_w, pad_w), mode='reflect')
        c = self.branch_col_conv(c)

        # 安全拼接
        fused = self.fuse(torch.cat([d, r, c], dim=1))
        fused = self.relu(self.norm(fused))

        out = x + self.res(fused)
        return torch.clamp(out, 0.0, 1.0)



class CrossConvNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 4, 4, 4], num_refinement_blocks=2,
                 bias=False, dilation_schedule=None):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if dilation_schedule is None:
            dilation_schedule = {i: 2 for i in range(sum(num_blocks) + num_refinement_blocks)}

        # 固定每层 kernel_size：第1层 30，第2层 15，第3层 7，第4层 3
        kernel_size_schedule_by_level = [31, 15, 7, 3]

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        count = 0
        for i in range(self.num_levels):
            level_dim = dim * (2 ** i)
            kernel_size = kernel_size_schedule_by_level[i]
            blocks = [CrossConvBlock(level_dim,
                                     dilation=dilation_schedule.get(count + j, 2),
                                     kernel_size=kernel_size,
                                     bias=bias)
                      for j in range(num_blocks[i])]
            count += num_blocks[i]
            self.encoders.append(nn.Sequential(*blocks))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        latent_dim = dim * (2 ** (self.num_levels - 1))
        self.latent = nn.Sequential(*[
            CrossConvBlock(latent_dim, dilation=2, kernel_size=3, bias=bias)
            for _ in range(num_blocks[-1])
        ])

        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            kernel_size = kernel_size_schedule_by_level[i]
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            blocks = [CrossConvBlock(out_dim,
                                     dilation=dilation_schedule.get(count + j, 2),
                                     kernel_size=kernel_size,
                                     bias=bias)
                      for j in range(num_blocks[i])]
            count += num_blocks[i]
            self.decoders.append(nn.Sequential(*blocks))

        self.refinement = nn.Sequential(*[
            CrossConvBlock(dim, dilation=dilation_schedule.get(count + i, 2), kernel_size=3, bias=bias)
            for i in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x_input = x
        feats = []
        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        x = self.latent(x)

        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)
            x = self.decoders[self.num_levels - 2 - i](x)

        x = self.refinement(x)
        return self.output(x) + x_input




    
class CrossConvRestormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[2, 2, 2, 2], num_refinement_blocks=2,
                 bias=False, dilation_schedule=None,
                 kernel_sizes=[31, 15, 7, 3]):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if dilation_schedule is None:
            dilation_schedule = {i: 2 for i in range(sum(num_blocks) + num_refinement_blocks)}

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        count = 0
        for i in range(self.num_levels):
            level_dim = dim * (2 ** i)
            ks = kernel_sizes[i]
            blocks = [CrossConvBlock(level_dim,
                                     dilation=dilation_schedule.get(count + j, 2),
                                     kernel_size=ks,
                                     bias=bias)
                      for j in range(num_blocks[i])]
            count += num_blocks[i]
            self.encoders.append(nn.Sequential(*blocks))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        latent_dim = dim * (2 ** (self.num_levels - 1))
        self.latent = nn.Sequential(*[
            CrossConvBlock(latent_dim, dilation=2, kernel_size=kernel_sizes[-1], bias=bias)
            for _ in range(num_blocks[-1])
        ])

        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            ks = kernel_sizes[i]
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))
            blocks = [CrossConvBlock(out_dim,
                                     dilation=dilation_schedule.get(count + j, 2),
                                     kernel_size=ks,
                                     bias=bias)
                      for j in range(num_blocks[i])]
            count += num_blocks[i]
            self.decoders.append(nn.Sequential(*blocks))

        self.refinement = nn.Sequential(*[
            CrossConvBlock(dim,
                           dilation=dilation_schedule.get(count + i, 2),
                           kernel_size=kernel_sizes[0],
                           bias=bias)
            for i in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        x_input = x
        feats = []
        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        x = self.latent(x)

        for idx, i in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[idx](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[idx](x)
            x = self.decoders[idx](x)

        x = self.refinement(x)
        return self.output(x) + x_input


import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 多尺度空洞卷积 Block ======
class MultiScaleDilatedConvBlock(nn.Module):
    def __init__(self, dim, dilations=[1, 2, 4], bias=False):
        super().__init__()
        # 多分支空洞卷积
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, 1, padding=d, dilation=d, bias=bias)
            for d in dilations
        ])
        # 融合 + 归一化
        self.fuse = nn.Conv2d(dim * len(dilations), dim, kernel_size=1, bias=bias)
        self.norm = nn.GroupNorm(1, dim)
        self.relu = nn.ReLU(inplace=True)
        # 残差卷积
        self.res = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(feats, dim=1))
        fused = self.relu(self.norm(fused))
        out = x + self.res(fused)
        return torch.clamp(out, 0.0, 1.0)


# ====== 新结构 DLConvNet ======
class DLConvNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 4, 4, 4], num_refinement_blocks=2,
                 bias=False):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 每个 Level 的多尺度空洞率设计
        dilation_sets_by_level = [
            [1, 5, 10, 15],  # Level 1
            [1, 3, 7],       # Level 2
            [1, 3],          # Level 3
            [1]              # Level 4
        ]

        # ====== Encoder ======
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_levels):
            level_dim = dim * (2 ** i)
            blocks = [MultiScaleDilatedConvBlock(level_dim,
                                                 dilations=dilation_sets_by_level[i],
                                                 bias=bias)
                      for _ in range(num_blocks[i])]
            self.encoders.append(nn.Sequential(*blocks))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        # ====== Latent ======
        latent_dim = dim * (2 ** (self.num_levels - 1))
        self.latent = nn.Sequential(*[
            MultiScaleDilatedConvBlock(latent_dim,
                                       dilations=[1, 3, 7],  # 可调整
                                       bias=bias)
            for _ in range(num_blocks[-1])
        ])

        # ====== Decoder ======
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            blocks = [MultiScaleDilatedConvBlock(out_dim,
                                                 dilations=dilation_sets_by_level[i],
                                                 bias=bias)
                      for _ in range(num_blocks[i])]
            self.decoders.append(nn.Sequential(*blocks))

        # ====== Refinement ======
        self.refinement = nn.Sequential(*[
            MultiScaleDilatedConvBlock(dim,
                                       dilations=[1, 3],  # 细节修正
                                       bias=bias)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x_input = x
        feats = []

        # Encoder
        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        # Latent
        x = self.latent(x)

        # Decoder
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)
            x = self.decoders[self.num_levels - 2 - i](x)

        # Refinement
        x = self.refinement(x)
        return self.output(x) + x_input



##############################################
# 轻量化多尺度 FFN（替代原 FFN）
##############################################
class LightMultiScaleFFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.0, bias=False):
        super().__init__()
        hidden = int(dim * expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)

        # 两个多尺度卷积分支（减少到 2 个）
        self.branch3 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=bias)
        self.branch_dil = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, bias=bias)

        # 融合
        self.fuse = nn.Conv2d(hidden * 2, hidden, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x_proj = self.project_in(x)
        b3 = F.gelu(self.branch3(x_proj))
        bd = F.gelu(self.branch_dil(x_proj))
        fused = self.fuse(torch.cat([b3, bd], dim=1))
        return self.project_out(fused)


##############################################
# 轻量化多尺度 Transformer Block
##############################################
class LightTransformerBlockMS(nn.Module):
    def __init__(self, dim, num_heads, expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = LightMultiScaleFFN(dim, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##############################################
# 轻量化 Restormer 多尺度版本
##############################################
class RestormerMS(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 4, 4, 4], num_refinement_blocks=1,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='BiasFree'):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 编码器
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_levels):
            level_dim = dim * (2 ** i)
            blocks = [LightTransformerBlockMS(level_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.encoders.append(nn.Sequential(*blocks))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        # 潜层
        self.latent = nn.Sequential(*[
            LightTransformerBlockMS(dim * (2 ** (self.num_levels - 1)), heads[-1],
                                    ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[-1])
        ])

        # 解码器
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))
            blocks = [LightTransformerBlockMS(out_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.decoders.append(nn.Sequential(*blocks))

        # 精修层
        self.refinement = nn.Sequential(*[
            LightTransformerBlockMS(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        x_input = x
        feats = []
        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)
        x = self.latent(x)
        for i in range(self.num_levels - 1)[::-1]:
            x = self.upsamples[self.num_levels - 2 - i](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[self.num_levels - 2 - i](x)
            x = self.decoders[self.num_levels - 2 - i](x)
        x = self.refinement(x)
        return self.output(x) + x_input


##############################################
# 跨尺度融合模块
##############################################
class CrossScaleFusion(nn.Module):
    def __init__(self, in_dims, out_dim, bias=False):
        """
        in_dims: list，不同尺度的输入通道数
        out_dim: 融合后的通道数
        """
        super().__init__()
        self.proj = nn.Conv2d(sum(in_dims), out_dim, kernel_size=1, bias=bias)

    def forward(self, feats, target_size):
        """
        feats: list of feature maps (多尺度特征)
        target_size: (H, W)，目标空间分辨率
        """
        resized_feats = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in feats]
        fused = torch.cat(resized_feats, dim=1)
        return self.proj(fused)


##############################################
# 跨尺度融合版 Transformer Block
##############################################
class TransformerBlockCSF(nn.Module):
    def __init__(self, dim, num_heads, expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##############################################
# Restormer 跨尺度融合版
##############################################
class RestormerCSF(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 4, 4, 4], num_refinement_blocks=2,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='BiasFree'):
        super().__init__()
        self.num_levels = len(num_blocks)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 编码器
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.num_levels):
            level_dim = dim * (2 ** i)
            blocks = [TransformerBlockCSF(level_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.encoders.append(nn.Sequential(*blocks))
            if i < self.num_levels - 1:
                self.downsamples.append(Downsample(level_dim))

        # 潜层
        self.latent = nn.Sequential(*[
            TransformerBlockCSF(dim * (2 ** (self.num_levels - 1)), heads[-1],
                                ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[-1])
        ])

        # 解码器 + 跨尺度融合
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.cross_fusions = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # 构建 CrossScaleFusion 时，需要知道所有 encoder 输出的通道
        encoder_dims = [dim * (2 ** i) for i in range(self.num_levels)]

        for i in reversed(range(self.num_levels - 1)):
            in_dim = dim * (2 ** (i + 1))
            out_dim = dim * (2 ** i)
            self.upsamples.append(Upsample(in_dim))
            self.reduce_chans.append(nn.Conv2d(in_dim, out_dim, 1, bias=bias))

            # 跨尺度融合模块
            self.cross_fusions.append(CrossScaleFusion(encoder_dims, out_dim, bias=bias))

            # 融合后的解码器
            blocks = [TransformerBlockCSF(out_dim, heads[i], ffn_expansion_factor, bias, LayerNorm_type)
                      for _ in range(num_blocks[i])]
            self.decoders.append(nn.Sequential(*blocks))

        # 精修层
        self.refinement = nn.Sequential(*[
            TransformerBlockCSF(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        x_input = x
        feats = []

        # 编码器
        x = self.patch_embed(x)
        for i in range(self.num_levels):
            x = self.encoders[i](x)
            feats.append(x)
            if i < self.num_levels - 1:
                x = self.downsamples[i](x)

        # 潜层
        x = self.latent(x)

        # 解码器 + 跨尺度融合
        for idx, i in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[idx](x)
            x = torch.cat([x, feats[i]], dim=1)
            x = self.reduce_chans[idx](x)

            # 跨尺度融合特征
            fusion_feat = self.cross_fusions[idx](feats, target_size=x.shape[2:])
            x = x + fusion_feat  # 残差融合

            x = self.decoders[idx](x)

        # 精修
        x = self.refinement(x)
        return self.output(x) + x_input
