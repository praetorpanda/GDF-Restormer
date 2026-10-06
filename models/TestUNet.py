import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic 2-layer Conv + ReLU block."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block: ConvBlock + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


class UpBlock(nn.Module):
    """Upsampling block: Upsample + ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Padding if necessary
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class TestUNet(nn.Module):
    """A simple UNet using only ConvBlocks."""
    def __init__(self, in_channels=3, out_channels=3):
        super(TestUNet, self).__init__()

        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)

        self.middle = ConvBlock(128, 256)

        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x = self.enc1(x)   # 64
        x2, x = self.enc2(x)   # 128


        x = self.middle(x)     # 512


        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        x = self.final(x)
        return x


if __name__ == "__main__":
    model = TestUNet(in_channels=3, out_channels=3)
    tensor = torch.randn(1, 3, 700, 350)
    out = model(tensor)
    print(f"Input shape: {tensor.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")