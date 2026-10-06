


# models/AdaIR_models.py
# AdaIR original model + AdaIR3 fair-scale variant
# Adapted for current project engine:
#   model(inp) -> pred
#   no internal clamp in forward
#   loss/metric clamp is handled by engine_data_prior.py

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Layer Norm
# ============================================================
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()

        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ============================================================
# GDFN
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim,
            hidden_features * 2,
            kernel_size=1,
            bias=bias,
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ============================================================
# MDTA
# ============================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=bias,
        )

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(
            q,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )

        k = rearrange(
            k,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )

        v = rearrange(
            v,
            "b (head c) h w -> b head c (h w)",
            head=self.num_heads,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=h,
            w=w,
        )

        out = self.project_out(out)
        return out


# ============================================================
# Resize modules
# ============================================================
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat,
                n_feat * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ============================================================
# Restormer Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
    ):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Channel-wise Cross Attention
# ============================================================
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super().__init__()

        self.num_head = num_head
        self.temperature = nn.Parameter(
            torch.ones(num_head, 1, 1),
            requires_grad=True,
        )

        self.q = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias,
        )

        self.q_dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=bias,
        )

        self.kv = nn.Conv2d(
            dim,
            dim * 2,
            kernel_size=1,
            bias=bias,
        )

        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x, y):
        assert x.shape == y.shape, (
            "The shape of feature maps from image and features are not equal! "
            f"x={x.shape}, y={y.shape}"
        )

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(
            q,
            "b (head c) h w -> b head c (h w)",
            head=self.num_head,
        )

        k = rearrange(
            k,
            "b (head c) h w -> b head c (h w)",
            head=self.num_head,
        )

        v = rearrange(
            v,
            "b (head c) h w -> b head c (h w)",
            head=self.num_head,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_head,
            h=h,
            w=w,
        )

        out = self.project_out(out)
        return out


# ============================================================
# Overlapped image patch embedding
# ============================================================
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()

        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        return self.proj(x)


# ============================================================
# H-L Unit
# ============================================================
class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

    def forward(self, x):
        max_map = torch.max(x, 1, keepdim=True)[0]
        mean_map = torch.mean(x, 1, keepdim=True)

        scale = torch.cat((max_map, mean_map), dim=1)
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)

        return scale


# ============================================================
# L-H Unit
# ============================================================
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super().__init__()

        hidden = max(dim // 16, 1)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max_v = self.mlp(self.max(x))

        scale = avg + max_v
        scale = torch.sigmoid(scale)

        return scale


# ============================================================
# Frequency Modulation Module
# ============================================================
class FreRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)

        self.proj = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
        )

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)

        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)

        return out


# ============================================================
# Adaptive Frequency Learning Block / FreModule
# ============================================================
class FreModule(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        bias,
        in_dim=3,
        fft_mask_n=128,
    ):
        super().__init__()

        self.fft_mask_n = int(fft_mask_n)

        self.conv1 = nn.Conv2d(
            in_dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # zero-init residual gate:
        # start as identity: out * 0 + y * 1
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(
            dim,
            num_head=num_heads,
            bias=bias,
        )

        self.channel_cross_h = Chanel_Cross_Attention(
            dim,
            num_head=num_heads,
            bias=bias,
        )

        self.channel_cross_agg = Chanel_Cross_Attention(
            dim,
            num_head=num_heads,
            bias=bias,
        )

        self.frequency_refine = FreRefine(dim)

        hidden = max(dim // 8, 1)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()

        x = F.interpolate(
            x,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        high_feature, low_feature = self.fft(x)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    @staticmethod
    def shift(x):
        b, c, h, w = x.shape
        return torch.roll(
            x,
            shifts=(int(h / 2), int(w / 2)),
            dims=(2, 3),
        )

    @staticmethod
    def unshift(x):
        b, c, h, w = x.shape
        return torch.roll(
            x,
            shifts=(-int(h / 2), -int(w / 2)),
            dims=(2, 3),
        )

    def fft(self, x):
        x = self.conv1(x)

        mask = torch.zeros_like(x)
        h, w = x.shape[-2:]

        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        n = self.fft_mask_n

        for i in range(mask.shape[0]):
            h_ = (h // n * threshold[i, 0, :, :]).int()
            w_ = (w // n * threshold[i, 1, :, :]).int()

            h_int = int(h_.item())
            w_int = int(w_.item())

            if h_int > 0 and w_int > 0:
                mask[
                    i,
                    :,
                    h // 2 - h_int:h // 2 + h_int,
                    w // 2 - w_int:w // 2 + w_int,
                ] = 1

        fft = torch.fft.fft2(
            x,
            norm="forward",
            dim=(-2, -1),
        )

        fft = self.shift(fft)

        fft_high = fft * (1 - mask)
        high = self.unshift(fft_high)
        high = torch.fft.ifft2(
            high,
            norm="forward",
            dim=(-2, -1),
        )
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(
            low,
            norm="forward",
            dim=(-2, -1),
        )
        low = torch.abs(low)

        return high, low


# ============================================================
# Original AdaIR 4-level model
# ============================================================
class AdaIR(nn.Module):
    """
    Original AdaIR-style 4-level model.

    Default:
        dim=48
        num_blocks=[4,6,6,8]
        refinement=4
        heads=[1,2,4,8]

    This version keeps original structure but adapts interface to:
        model(inp) -> pred

    No internal clamp.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        decoder=True,
    ):
        super().__init__()

        assert len(num_blocks) == 4
        assert len(heads) == 4

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)
        self.decoder = decoder

        if self.decoder:
            self.fre1 = FreModule(
                dim * 2 ** 3,
                num_heads=heads[2],
                bias=bias,
            )

            self.fre2 = FreModule(
                dim * 2 ** 2,
                num_heads=heads[2],
                bias=bias,
            )

            self.fre3 = FreModule(
                dim * 2 ** 1,
                num_heads=heads[2],
                bias=bias,
            )

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 4,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 8,
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(
            dim * 8,
            dim * 4,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 4,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(dim * 4)

        self.reduce_chan_level2 = nn.Conv2d(
            dim * 4,
            dim * 2,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            dim * 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, inp_img, noise_emb=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            latent = self.fre1(inp_img, latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat(
            [inp_dec_level3, out_enc_level3],
            dim=1,
        )
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2],
            dim=1,
        )
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1,
        )

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out = self.output(out_dec_level1) + inp_img

        return out


# ============================================================
# AdaIR3 fair-scale model
# ============================================================
class AdaIR3(nn.Module):
    """
    3-level AdaIR for fair comparison with current 3-level Restormer setting.

    Backbone:
        L1: dim
        L2: dim*2
        L3: dim*4

    Frequency modules:
        fre1: latent, dim*4
        fre2: decoder level2, dim*2

    Interface:
        model(inp) -> pred

    No internal clamp.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=(2, 3, 3),
        num_refinement_blocks=1,
        heads=(1, 2, 4),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="BiasFree",
        decoder=True,
    ):
        super().__init__()

        assert len(num_blocks) == 3
        assert len(heads) == 3

        self.decoder = decoder

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(dim * 2)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 4,
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        if self.decoder:
            self.fre1 = FreModule(
                dim * 4,
                num_heads=heads[2],
                bias=bias,
            )

            self.fre2 = FreModule(
                dim * 2,
                num_heads=heads[1],
                bias=bias,
            )

        self.up3_2 = Upsample(dim * 4)

        self.reduce_chan_level2 = nn.Conv2d(
            dim * 4,
            dim * 2,
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(dim * 2)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 2,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            dim * 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, inp_img, noise_emb=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        latent = self.latent(inp_enc_level3)

        if self.decoder:
            latent = self.fre1(inp_img, latent)

        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2],
            dim=1,
        )
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.fre2(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1,
        )

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out = self.output(out_dec_level1) + inp_img

        return out