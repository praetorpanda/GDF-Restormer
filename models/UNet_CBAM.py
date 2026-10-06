import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2  # 保证输出尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))



class CBAMBlock(nn.Module):
    def __init__(self, in_ch):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlockCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CBAMBlock(out_ch)
        )

    def forward(self, x):
        return self.block(x)


class UNet_CBAM(nn.Module):
    def __init__(self, in_ch=6, base_ch=64, out_ch=3):
        super().__init__()
        self.enc1 = ConvBlockCBAM(in_ch, base_ch)
        self.enc2 = ConvBlockCBAM(base_ch, base_ch * 2)
        self.enc3 = ConvBlockCBAM(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlockCBAM(base_ch * 4, base_ch * 8)

        self.dec3 = ConvBlockCBAM(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = ConvBlockCBAM(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlockCBAM(base_ch * 2 + base_ch, base_ch)

        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out_conv(d1)
