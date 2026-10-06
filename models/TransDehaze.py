# models/TransDehaze.py
import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# 基础模块：FFN / Attention / Encoder / Decoder
# （完全按官方实现，不改结构）
# ============================================================

class Ffn(nn.Module):
    # feed forward network layer after attention
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # NOTE: 官方实现中 forward 就是标准的两层 MLP
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE: scale factor 与官方一致
        self.scale = qk_scale or head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        N, L, D = q.shape
        q, k, v = self.query(q), self.key(k), self.value(v)
        q = q.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(N, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, pos):
        # 官方实现：直接 x+pos 进 attention，不做 pre-norm 的 x+pos 再 norm
        q, k, v = x + pos, x + pos, x
        x = x + self.attn(q, k, v)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        self.attn2 = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm3 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = Ffn(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, pos, task_embed):
        memory = x
        x = self.norm1(x)
        q, k, v = x + task_embed, x + task_embed, x
        x = x + self.attn1(q, k, v)
        x = self.norm2(x)
        q, k, v = x + task_embed, memory + pos, memory
        x = x + self.attn2(q, k, v)
        x = x + self.ffn(self.norm3(x))
        return x


# ============================================================
# CNN 部分：Head / ResBlock / PatchEmbed / DePatchEmbed / Tail
# （同官方）
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class Head(nn.Module):
    """ 与官方相同的 Head：Conv + 两个 ResBlock """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.resblock2(out)
        return out


class PatchEmbed(nn.Module):
    """ Feature -> Patch Embedding（官方版本） """

    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x):
        N, C, H, W = ori_shape = x.shape
        p = self.patch_size
        num_patches = (H // p) * (W // p)
        out = torch.zeros((N, num_patches, self.dim), device=x.device, dtype=x.dtype)
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, k, :] = x[:, :, i:i + p, j:j + p].flatten(1)
            i += p
        return out, ori_shape


class DePatchEmbed(nn.Module):
    """ Patch Embedding -> Feature（官方版本） """

    def __init__(self, patch_size=1, in_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = self.patch_size ** 2 * in_channels

    def forward(self, x, ori_shape):
        N, num_patches, dim = x.shape
        _, C, H, W = ori_shape
        p = self.patch_size
        out = torch.zeros(ori_shape, device=x.device, dtype=x.dtype)
        i, j = 0, 0
        for k in range(num_patches):
            if i + p > W:
                i = 0
                j += p
            out[:, :, i:i + p, j:j + p] = x[:, k, :].reshape(N, C, p, p)
            i += p
        return out


class Tail(nn.Module):
    """ 官方 Tail：SR 任务 + 普通 Conv 输出 RGB """

    def __init__(self, task_id, in_channels, out_channels):
        super().__init__()
        assert 0 <= task_id <= 5
        # 0,1: noise; 2,3,4: SR x2,x3,x4; 5: dehaze
        upscale_map = [1, 1, 2, 3, 4, 1]
        scale = upscale_map[task_id]
        m = []
        if scale > 1:
            m.append(nn.Conv2d(
                in_channels, in_channels * scale * scale,
                kernel_size=3, stride=1, padding=1, bias=False
            ))
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    m.append(nn.PixelShuffle(2))
            elif scale == 3:
                m.append(nn.PixelShuffle(3))
            else:
                raise NameError("Only support x3 and x2^n SR")

        m.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        ))
        self.m = nn.Sequential(*m)

    def forward(self, x):
        return self.m(x)


# ============================================================
# Truncated Normal 初始化（官方）
# ============================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on
    # https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # 标准正态 CDF
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor, mean=0., std=1., a=-2., b=2.) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================================================
# ImageProcessingTransformer（官方 IPT / TransDehaze 核心）
# ============================================================

class ImageProcessingTransformer(nn.Module):
    """ 官方 IPT 主干。 """

    def __init__(self,
                 patch_size=1,
                 in_channels=3,
                 mid_channels=64,
                 num_classes=1000,
                 depth=12,
                 num_heads=8,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.task_id = None
        self.num_classes = num_classes
        self.embed_dim = patch_size * patch_size * mid_channels

        self.headsets = nn.ModuleList([Head(in_channels, mid_channels) for _ in range(6)])
        self.headsets2 = nn.ModuleList([Head(mid_channels, mid_channels) for _ in range(6)])
        self.headsets3 = nn.ModuleList([Head(mid_channels, mid_channels) for _ in range(6)])

        self.patch_embedding = PatchEmbed(patch_size=patch_size, in_channels=mid_channels)
        self.embed_dim = self.patch_embedding.dim

        # 官方 CA：输入通道数固定为 3 * 64 = 192
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 64 // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 16, 192, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        if self.embed_dim % num_heads != 0:
            raise RuntimeError("Embedding dim must be devided by numbers of heads")

        # 位置编码 / 任务编码，假定输入为 48×48
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (48 // patch_size) ** 2, self.embed_dim)
        )
        self.task_embed = nn.Parameter(
            torch.zeros(6, 1, (48 // patch_size) ** 2, self.embed_dim)
        )

        self.encoder = nn.ModuleList([
            EncoderLayer(
                dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(
                dim=self.embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        self.de_patch_embedding = DePatchEmbed(patch_size=patch_size, in_channels=mid_channels)
        self.tailsets = nn.ModuleList(
            [Tail(id, mid_channels, in_channels) for id in range(6)]
        )

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def set_task(self, task_id: int):
        self.task_id = task_id

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # ===== 官方实现这里直接强制 task_id=5（dehaze） =====
        self.task_id = 5
        assert 0 <= self.task_id <= 5

        # 三路 Head
        x1 = self.headsets[self.task_id](x)
        x2 = self.headsets2[self.task_id](x1)
        x3 = self.headsets3[self.task_id](x2)

        # Patch Embedding
        x1_tokens, ori_shape1 = self.patch_embedding(x1)
        x2_tokens, ori_shape2 = self.patch_embedding(x2)
        x3_tokens, ori_shape3 = self.patch_embedding(x3)

        # Encoder
        pos = self.pos_embed[:, :x1_tokens.shape[1]]
        for blk in self.encoder:
            x1_tokens = blk(x1_tokens, pos)
            x2_tokens = blk(x2_tokens, pos)
            x3_tokens = blk(x3_tokens, pos)

        # Decoder with task embedding
        task_pos = self.task_embed[self.task_id, :, :x1_tokens.shape[1]]
        for blk in self.decoder:
            x1_tokens = blk(x1_tokens, pos, task_pos)
            x2_tokens = blk(x2_tokens, pos, task_pos)
            x3_tokens = blk(x3_tokens, pos, task_pos)

        # DePatch
        x1_rec = self.de_patch_embedding(x1_tokens, ori_shape1)
        x2_rec = self.de_patch_embedding(x2_tokens, ori_shape2)
        x3_rec = self.de_patch_embedding(x3_tokens, ori_shape3)

        # CA 融合
        cat = torch.cat([x1_rec, x2_rec, x3_rec], dim=1)
        w = self.ca(cat)                      # (N, 192, 1, 1)
        w = w.view(-1, 3, 64)[:, :, :, None, None]  # (N,3,64,1,1)

        out = w[:, 0, :, :, :] * x1_rec + \
              w[:, 1, :, :, :] * x2_rec + \
              w[:, 2, :, :, :] * x3_rec

        out = self.tailsets[self.task_id](out)
        return out


# ============================================================
# 工厂函数：TD_base（官方 TransDehaze / IPT Base 设置）
# ============================================================

def TD_base(**kwargs):
    """
    严格对齐官方设置：
      - patch_size=4
      - depth=1
      - num_heads=1
      - ffn_ratio=4
      - qkv_bias=True
      - LayerNorm eps=1e-6
    """
    model = ImageProcessingTransformer(
        patch_size=4,
        depth=1,
        num_heads=1,
        ffn_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# ============================================================
# Wrapper：适配你现有训练主程序（严格公平版本）
# ============================================================

class TransDehaze(nn.Module):
    """
    严格公平版 Wrapper：
      - 不改 IPT 结构，不改 Encoder/Decoder，不删 CA、不加重叠窗口
      - 只做两件事：
          1) [0,1] -> [-1,1] 再喂给 IPT（与官方训练脚本一致）
          2) 任意尺寸 (B,3,H,W)：整体 resize 到 48x48 再跑 IPT，
             输出再 resize 回原尺寸。
    """

    def __init__(self, in_channels=3, out_channels=3,
                 base_image_size=48):
        super().__init__()
        assert in_channels == 3 and out_channels == 3, \
            "严格公平版：保持官方 RGB 输入输出 (3 通道)"

        self.base_size = base_image_size
        self.core = TD_base(
            in_channels=in_channels,
            mid_channels=64,
            num_classes=1000
        )

    def forward(self, x):
        """
        x: (B,3,H,W), 值域为 [0,1]（与你现有 pipeline 一致）
        """
        B, C, H, W = x.shape

        # === 1. 对齐官方 Normalize： [0,1] -> [-1,1] ===
        # 官方训练：ToTensor + Normalize(mean=0.5,std=0.5)
        # 即： (x - 0.5) / 0.5 = 2x - 1
        x_norm = (x - 0.5) / 0.5

        # === 2. resize 到 48x48（官方 IPT 假定输入 48x48） ===
        base = self.base_size
        if H != base or W != base:
            x_resized = F.interpolate(
                x_norm, size=(base, base),
                mode="bilinear", align_corners=False
            )
        else:
            x_resized = x_norm

        # === 3. IPT 主干（严格官方实现） ===
        y_resized = self.core(x_resized)  # 输出同样是 [-1,1] 语义

        # === 4. resize 回原尺寸 ===
        if H != base or W != base:
            y_norm = F.interpolate(
                y_resized, size=(H, W),
                mode="bilinear", align_corners=False
            )
        else:
            y_norm = y_resized

        # === 5. 反归一化回 [0,1]，与其他模型对齐 ===
        y = y_norm * 0.5 + 0.5
        y = torch.clamp(y, 0.0, 1.0)

        return y
