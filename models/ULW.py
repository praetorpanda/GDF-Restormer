import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        # Encoder (Downsampling)
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Decoder (Upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))
        
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoding
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoding
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.sigmoid(self.final_conv(x))
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    

class LearnableWienerLayer(nn.Module):
    def __init__(self, channels=3, kernel_size=5):
        super(LearnableWienerLayer, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Depthwise convolution: one kernel per channel
        self.filter = nn.Conv2d(
            channels, channels, kernel_size=kernel_size,
            groups=channels, bias=False, padding=self.padding
        )

        # Initialize filter to Gaussian blur (Wiener-like)
        for name, param in self.filter.named_parameters():
            if "weight" in name:
                with torch.no_grad():
                    param.copy_(self._init_gaussian_kernel(channels, kernel_size))

        # Learnable noise level (denominator for Wiener-like control)
        self.noise_power = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        # Smooth signal
        signal = self.filter(x)

        # Simulate Wiener-like effect: weighted mix
        # output = signal * (signal^2 / (signal^2 + noise_power))
        eps = 1e-6
        power = signal ** 2
        wiener_response = power / (power + self.noise_power.view(1, -1, 1, 1) + eps)
        return signal * wiener_response

    def _init_gaussian_kernel(self, channels, kernel_size):
        import numpy as np
        import math

        def gaussian_2d(size, sigma=1.0):
            ax = np.arange(-size // 2 + 1., size // 2 + 1.)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            return kernel / np.sum(kernel)

        kernel = gaussian_2d(kernel_size, sigma=1.0)
        kernel = torch.from_numpy(kernel).float()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel.repeat(channels, 1, 1, 1)


class UNetWithWiener(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetWithWiener, self).__init__()
        self.wiener = LearnableWienerLayer(out_channels)

        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.wiener(x)
        x = self.unet(x)
        # x = self.unet(x)
        # x = self.wiener(x)
        return x
    
class ULW(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, wiener_kernel=5):
        super().__init__()
        
        self.wiener = LearnableWienerLayer(
            channels=in_channels,
            kernel_size=wiener_kernel
        )

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x_w = self.wiener(x)   # Wiener 去烟
        out = self.unet(x_w)   # UNet 重建
        return out
    

    
# model = LearnableWienerLayer()
# model = model.cuda()
# summary(model, input_size = (3, 700, 350))

