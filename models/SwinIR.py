import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# LayerNorm2D for Conv input
# -----------------------------
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# -----------------------------
# Window-based Multi-head Self-Attention (W-MSA)
# -----------------------------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, heads=6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

# -----------------------------
# Swin Transformer Block
# -----------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        # Step 1: BCHW → BHW*C → window partition
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x_windows = window_partition(x.permute(0, 3, 1, 2), self.window_size)  # B*, W, W, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Step 2: attention + MLP
        attn_out = self.attn(self.norm1(x_windows))
        mlp_out = self.mlp(self.norm2(attn_out))
        x_windows = x_windows + attn_out + mlp_out

        # Step 3: window reverse
        x = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # B, C, H, W

        return shortcut + x

# -----------------------------
# SwinIRRefineNet
# -----------------------------
class SwinIRRefineNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, dim=60, num_blocks=2, window_size=8):
        super().__init__()
        self.embed = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[
            SwinBlock(dim, window_size=window_size) for _ in range(num_blocks)
        ])
        self.output = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.embed(x)
        feat = self.body(feat)
        out = self.output(feat)
        return out
