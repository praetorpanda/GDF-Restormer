"""
Restormer variants for the current metalens restoration project.

This file provides two model classes:

1. Restormer
   - Flexible N-level Restormer-like implementation.
   - Supports 2-level / 3-level / 4-level configurations.
   - Topology is corrected to match official Restormer logic:
       encoder levels = level 1 to level N-1
       latent level   = level N
       decoder level1 keeps 2*dim channels after concatenation.
   - Useful for your 3-level experiments.

2. RestormerOfficial4LightFit
   - Explicit 4-level official Restormer topology.
   - Intended as the paper-level 4-layer Restormer baseline.
   - Same official structure, but with memory-feasible defaults:
       dim=32
       num_blocks=[2,3,3,4]
       num_refinement_blocks=1
       heads=[1,2,4,8]
       LayerNorm_type="WithBias"
   - No FreModule, no coord, no prior, no clamp inside forward.

Official Restormer:
"Restormer: Efficient Transformer for High-Resolution Image Restoration"
Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat,
Fahad Shahbaz Khan, and Ming-Hsuan Yang
https://arxiv.org/abs/2111.09881
"""

import os
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# Backward-compatible debug helper
# ============================================================
# Some existing project modules, such as Restormer_PriorVariants.py,
# import maybe_debug_tensor from models.Restormer.
# Keep this function for compatibility.
# It is disabled by default and does not affect official Restormer behavior.
DEBUG_RESTORMER = os.environ.get("RESTORMER_DEBUG", "0") == "1"
DEBUG_PROB = float(os.environ.get("RESTORMER_DEBUG_PROB", 0.05))


def maybe_debug_tensor(name, tensor):
    if DEBUG_RESTORMER and torch.rand(1).item() < DEBUG_PROB:
        try:
            cpu_tensor = tensor.detach().float().cpu()
            if torch.isnan(cpu_tensor).any() or torch.isinf(cpu_tensor).any():
                print(
                    f"[DEBUG] {name} has NaN or Inf: "
                    f"min={cpu_tensor.min().item():.4f}, "
                    f"max={cpu_tensor.max().item():.4f}, "
                    f"shape={tuple(tensor.shape)}"
                )
            else:
                print(
                    f"[DEBUG] {name}: "
                    f"min={cpu_tensor.min().item():.4f}, "
                    f"max={cpu_tensor.max().item():.4f}, "
                    f"mean={cpu_tensor.mean().item():.4f}, "
                    f"shape={tuple(tensor.shape)}"
                )
        except Exception as e:
            print(f"[DEBUG] Failed to debug {name}: {e}")

##########################################################################
# Layer Norm helpers
##########################################################################

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()

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
        super(WithBias_LayerNorm, self).__init__()

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
        super(LayerNorm, self).__init__()

        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
# Gated-Dconv Feed-Forward Network, GDFN
##########################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

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


##########################################################################
# Multi-DConv Head Transposed Self-Attention, MDTA
##########################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, stable_softmax=False):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.stable_softmax = stable_softmax

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

        # Official Restormer uses direct softmax.
        # stable_softmax=True only subtracts a constant before softmax,
        # which is mathematically equivalent but may improve numerical stability.
        if self.stable_softmax:
            attn = attn - attn.amax(dim=-1, keepdim=True)

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


##########################################################################
# Transformer Block
##########################################################################

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        stable_softmax=False,
    ):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
            stable_softmax=stable_softmax,
        )

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
# Overlapped image patch embedding with 3x3 Conv
##########################################################################

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
# Resizing modules
##########################################################################

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

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
        super(Upsample, self).__init__()

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


##########################################################################
# Flexible official-topology Restormer
##########################################################################

class Restormer(nn.Module):
    """
    Flexible Restormer with official topology.

    Difference from the previous dynamic implementation:
    - The deepest level is latent only, not both encoder and latent.
    - The last decoder level does not use a 1x1 channel reduction.
    - decoder_level1 / refinement / output operate on 2*dim channels,
      matching official Restormer.

    Usage examples:
        3-level:
            dim=32,
            num_blocks=[2,3,3],
            heads=[1,2,4],
            num_refinement_blocks=1

        4-level:
            dim=32,
            num_blocks=[2,3,3,4],
            heads=[1,2,4,8],
            num_refinement_blocks=1
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
        dual_pixel_task=False,
        stable_softmax=False,
    ):
        super(Restormer, self).__init__()

        assert len(num_blocks) == len(heads), (
            f"num_blocks and heads must have the same length, "
            f"got len(num_blocks)={len(num_blocks)}, len(heads)={len(heads)}"
        )
        assert len(num_blocks) >= 2, "Restormer requires at least 2 levels."

        self.num_levels = len(num_blocks)
        self.dim = dim
        self.dual_pixel_task = dual_pixel_task

        self.patch_embed = OverlapPatchEmbed(
            inp_channels,
            dim,
            bias=bias,
        )

        # Encoder levels: level 1 to level N-1.
        # The deepest level N is latent, not encoder.
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for level in range(self.num_levels - 1):
            level_dim = int(dim * 2 ** level)

            self.encoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=level_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )

            self.downsamples.append(Downsample(level_dim))

        # Latent level: level N.
        latent_dim = int(dim * 2 ** (self.num_levels - 1))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=latent_dim,
                    num_heads=heads[-1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[-1])
            ]
        )

        # Decoder levels: from level N-1 back to level 1.
        self.upsamples = nn.ModuleList()
        self.reduce_chans = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for level in reversed(range(self.num_levels - 1)):
            high_dim = int(dim * 2 ** (level + 1))
            skip_dim = int(dim * 2 ** level)

            self.upsamples.append(Upsample(high_dim))

            concat_dim = skip_dim * 2

            if level > 0:
                # Official Restormer reduces channels for Level 3 and Level 2.
                self.reduce_chans.append(
                    nn.Conv2d(
                        concat_dim,
                        skip_dim,
                        kernel_size=1,
                        bias=bias,
                    )
                )
                decoder_dim = skip_dim
            else:
                # Official Restormer does NOT reduce channels at Level 1.
                self.reduce_chans.append(nn.Identity())
                decoder_dim = concat_dim

            self.decoders.append(
                nn.Sequential(
                    *[
                        TransformerBlock(
                            dim=decoder_dim,
                            num_heads=heads[level],
                            ffn_expansion_factor=ffn_expansion_factor,
                            bias=bias,
                            LayerNorm_type=LayerNorm_type,
                            stable_softmax=stable_softmax,
                        )
                        for _ in range(num_blocks[level])
                    ]
                )
            )

        refinement_dim = int(dim * 2)

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=refinement_dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim,
                refinement_dim,
                kernel_size=1,
                bias=bias,
            )

        self.output = nn.Conv2d(
            refinement_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, inp_img):
        feats = []

        x = self.patch_embed(inp_img)

        for level in range(self.num_levels - 1):
            x = self.encoders[level](x)
            feats.append(x)
            x = self.downsamples[level](x)

        x = self.latent(x)

        for decoder_index, level in enumerate(reversed(range(self.num_levels - 1))):
            x = self.upsamples[decoder_index](x)
            x = torch.cat([x, feats[level]], dim=1)
            x = self.reduce_chans[decoder_index](x)
            x = self.decoders[decoder_index](x)

        x = self.refinement(x)

        if self.dual_pixel_task:
            x = x + self.skip_conv(feats[0])
            x = self.output(x)
        else:
            x = self.output(x) + inp_img

        return x


##########################################################################
# Official 4-level Restormer, explicit implementation
##########################################################################

class RestormerOfficial4(nn.Module):
    """
    Explicit 4-level official Restormer implementation.

    This class follows the official restormer_arch.py topology:
        patch_embed
        encoder_level1
        down1_2
        encoder_level2
        down2_3
        encoder_level3
        down3_4
        latent
        up4_3
        reduce_chan_level3
        decoder_level3
        up3_2
        reduce_chan_level2
        decoder_level2
        up2_1
        decoder_level1
        refinement
        output + residual

    The default arguments below are the official Restormer defaults.
    For memory-feasible experiments, use RestormerOfficial4LightFit.
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
        dual_pixel_task=False,
        stable_softmax=False,
    ):
        super(RestormerOfficial4, self).__init__()

        assert len(num_blocks) == 4, "RestormerOfficial4 requires 4 block numbers."
        assert len(heads) == 4, "RestormerOfficial4 requires 4 head numbers."

        self.patch_embed = OverlapPatchEmbed(
            inp_channels,
            dim,
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
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2 ** 3))

        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2 ** 3),
            int(dim * 2 ** 2),
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 ** 2),
            int(dim * 2 ** 1),
            kernel_size=1,
            bias=bias,
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        # Official detail:
        # From Level 2 to Level 1, there is NO 1x1 conv to reduce channels.
        # After concatenation, decoder_level1 uses 2*dim channels.
        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2 ** 1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    stable_softmax=stable_softmax,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.dual_pixel_task = dual_pixel_task

        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(
                dim,
                int(dim * 2 ** 1),
                kernel_size=1,
                bias=bias,
            )

        self.output = nn.Conv2d(
            int(dim * 2 ** 1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat(
            [inp_dec_level3, out_enc_level3],
            dim=1,
        )
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2],
            dim=1,
        )
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1],
            dim=1,
        )
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1


class RestormerOfficial4LightFit(RestormerOfficial4):
    """
    Official 4-level Restormer topology with memory-feasible defaults.

    Use this as the paper-level 4-layer Restormer baseline under
    the current GPU constraint.

    It preserves:
        - official 4-level topology
        - MDTA
        - GDFN
        - PixelUnshuffle / PixelShuffle down-up sampling
        - no 1x1 reduction at decoder level1
        - refinement stage
        - residual output

    It compromises only:
        - dim: 48 -> 32
        - num_blocks: [4,6,6,8] -> [2,3,3,4]
        - num_refinement_blocks: 4 -> 1
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=(2, 3, 3, 4),
        num_refinement_blocks=1,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
        stable_softmax=False,
    ):
        super(RestormerOfficial4LightFit, self).__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            stable_softmax=stable_softmax,
        )


# Optional aliases for easier importing in run files.
Restormer4Official = RestormerOfficial4
Restormer4OfficialLightFit = RestormerOfficial4LightFit