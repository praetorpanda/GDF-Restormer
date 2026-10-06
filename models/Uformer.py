import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
import math

# ========= Feed Forward =========
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


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)
        return x


# ========= Projections =========
class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B, N, C = x.shape
        attn_kv = attn_kv if attn_kv is not None else x
        q = self.to_q(x).reshape(B, N, self.heads, -1).permute(0, 2, 1, 3)
        kv = self.to_kv(attn_kv).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return q, kv[0], kv[1]


# ========= Attention =========
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim, num_heads, head_dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# ========= Window Ops =========
def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows

def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ========= Transformer Block =========
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(win_size), num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        x_windows = window_partition(x, self.win_size).view(-1, self.win_size * self.win_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        x = window_reverse(attn_windows, self.win_size, H, W).view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ========= Uformer Layer =========
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim, input_resolution, num_heads, win_size,
                                  shift_size=0 if i % 2 == 0 else win_size // 2,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path if isinstance(drop_path, float) else drop_path[i],
                                  norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ========= Projections =========
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 4, 2, 1)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.conv(x).flatten(2).transpose(1, 2)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, 2, 2)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        return self.deconv(x).flatten(2).transpose(1, 2)


# ========= Uformer 主体 =========
class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.dd_in = dd_in
        self.reso = img_size

        # Input / Output
        self.input_proj = InputProj(dd_in, embed_dim)
        self.output_proj = OutputProj(2 * embed_dim, in_chans)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(embed_dim, (img_size, img_size), depths[0], num_heads[0], win_size)
        self.dowsample_0 = Downsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(embed_dim * 2, (img_size // 2, img_size // 2), depths[1], num_heads[1], win_size)
        self.dowsample_1 = Downsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(embed_dim * 4, (img_size // 4, img_size // 4), depths[2], num_heads[2], win_size)
        self.dowsample_2 = Downsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(embed_dim * 8, (img_size // 8, img_size // 8), depths[3], num_heads[3], win_size)
        self.dowsample_3 = Downsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(embed_dim * 16, (img_size // 16, img_size // 16), depths[4], num_heads[4], win_size)

        # Decoder
        self.upsample_0 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(embed_dim * 16, (img_size // 8, img_size // 8), depths[5], num_heads[5], win_size)
        self.upsample_1 = Upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(embed_dim * 8, (img_size // 4, img_size // 4), depths[6], num_heads[6], win_size)
        self.upsample_2 = Upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(embed_dim * 4, (img_size // 2, img_size // 2), depths[7], num_heads[7], win_size)
        self.upsample_3 = Upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(embed_dim * 2, (img_size, img_size), depths[8], num_heads[8], win_size)

    def forward(self, x):
        y = self.input_proj(x)
        conv0 = self.encoderlayer_0(y); pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0); pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1); pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2); pool3 = self.dowsample_3(conv3)

        conv4 = self.conv(pool3)

        up0 = self.upsample_0(conv4); deconv0 = self.decoderlayer_0(torch.cat([up0, conv3], -1))
        up1 = self.upsample_1(deconv0); deconv1 = self.decoderlayer_1(torch.cat([up1, conv2], -1))
        up2 = self.upsample_2(deconv1); deconv2 = self.decoderlayer_2(torch.cat([up2, conv1], -1))
        up3 = self.upsample_3(deconv2); deconv3 = self.decoderlayer_3(torch.cat([up3, conv0], -1))

        y = self.output_proj(deconv3)
        return x + y if self.dd_in == 3 else y
