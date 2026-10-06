import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
#           Attention 模块
# ===========================================
class ComplexAttention(nn.Module):
    def __init__(self, dim):
        super(ComplexAttention, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_attn(x) * self.spatial_attn(x)


# ===========================================
#         多尺度卷积分支
# ===========================================
class MultiScaleFusion(nn.Module):
    def __init__(self, dim):
        super(MultiScaleFusion, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3),
            nn.PReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.fusion(out)

        return x + out


# ===========================================
#          三尺度空洞卷积
# ===========================================
class TriScaleConv(nn.Module):
    def __init__(self, in_channels, outchannel=None):
        super(TriScaleConv, self).__init__()
        outchannel = outchannel if outchannel is not None else in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, outchannel, 3, dilation=5, padding=5),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, outchannel, 3, padding=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, outchannel, 3, dilation=3, padding=3),
            nn.PReLU()
        )

        self.merge = nn.Sequential(
            nn.Conv2d(outchannel * 4, outchannel * 2, 1),
            nn.GELU(),
            nn.Conv2d(outchannel * 2, outchannel, 1)
        )

    def forward(self, x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2

        out = torch.cat([x, x1, x2, x3], dim=1)
        out = self.merge(out)

        return x + out


# ===========================================
#            DWConv 模块
# ===========================================
class SimpleDWConv(nn.Module):
    def __init__(self, dim):
        super(SimpleDWConv, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


# ===========================================
#              双尺度卷积
# ===========================================
class DualScaleConv(nn.Module):
    def __init__(self, in_channels, outchannel):
        super(DualScaleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, outchannel, 3, dilation=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels, outchannel, 3, padding=1)

        self.merge = nn.Sequential(
            nn.Conv2d(outchannel, outchannel * 2, 1),
            nn.GELU(),
            nn.Conv2d(outchannel * 2, outchannel, 1)
        )

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x)
        out = self.merge(out)
        return x + out


# ===========================================
#         ComplexStructureBlock（核心）
# ===========================================
class ComplexStructureBlock(nn.Module):
    def __init__(self, dim):
        super(ComplexStructureBlock, self).__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = ComplexAttention(dim)
        self.multi_scale = MultiScaleFusion(dim)
        self.tri_scale = TriScaleConv(dim)
        self.dw = SimpleDWConv(dim)
        self.dual = DualScaleConv(dim, dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.attn(x)
        x = self.multi_scale(x)
        x = self.tri_scale(x)
        x = self.dw(x)
        x = self.dual(x)
        x = self.mlp(x)

        return x + identity


# ===========================================
#          ComplexDehazeNet（U-Net）
# ===========================================
class ComplexDehazeNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 embed_dims=[32, 64, 128],
                 depths=[8, 12, 16]):
        super(ComplexDehazeNet, self).__init__()

        self.patch_embed = nn.Conv2d(in_channels, embed_dims[0], kernel_size=3, padding=1)

        self.layers = nn.ModuleList([
            nn.Sequential(*[ComplexStructureBlock(embed_dims[i]) for _ in range(depths[i])])
            for i in range(len(embed_dims))
        ])

        self.transition = nn.ModuleList([
            nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, stride=2, padding=1)
            for i in range(len(embed_dims) - 1)
        ])

        self.upscale = nn.ModuleList([
            nn.ConvTranspose2d(embed_dims[i + 1], embed_dims[i], 3, stride=2, padding=1, output_padding=1)
            for i in range(len(embed_dims) - 1)
        ])

        self.skip_adapters = nn.ModuleList([
            nn.Conv2d(embed_dims[i], embed_dims[i], 1) for i in range(len(embed_dims) - 1)
        ])

        self.final_conv = nn.Conv2d(embed_dims[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.patch_embed(x)
        skips = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                skips.append(x)
                x = self.transition[i](x)

        for i in range(len(skips) - 1, -1, -1):
            x = self.upscale[i](x)
            x = x + self.skip_adapters[i](skips[i])

        return self.final_conv(x)


# ===========================================
#      工厂函数：与主程序保持一致
# ===========================================
def complexdehazenet_s():
    return ComplexDehazeNet(
        in_channels=3,
        out_channels=3,
        embed_dims=[32, 64, 128],
        depths=[8, 12, 16]
    )
