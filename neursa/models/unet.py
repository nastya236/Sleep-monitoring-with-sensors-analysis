import torch

from torch import nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_2d=False):
        super().__init__()
        if use_2d:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_2d=False):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels, kernel_size, use_2d)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, use_2d=False):
        super().__init__()
        if use_2d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_2d=False):
        super().__init__()
        if use_2d:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConvBlock(in_channels, out_channels, kernel_size, use_2d)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool1d(2),
                DoubleConvBlock(in_channels, out_channels, kernel_size, use_2d)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_2d=False, linear=False):
        super().__init__()
        self.use_2d = use_2d

        #  would be a nice idea if the upsampling could be learned too
        if linear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            if not use_2d:
                self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, 2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = DoubleConvBlock(in_channels, out_channels, kernel_size, use_2d)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if not self.use_2d:
            assert x1.dim() == 3
            diffY = x2.size()[-1] - x1.size()[-1]
            x1 = nn.functional.pad(x1, (diffY // 2, diffY - diffY // 2))
        else:
            diffY = x2.size()[-2] - x1.size()[-2]
            diffX = x2.size()[-1] - x1.size()[-1]
            x1 = nn.functional.pad(x1, (diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.inc = InConvBlock(channels[0], channels[1])
        self.down1 = DownBlock(channels[1], channels[2])
        self.down2 = DownBlock(channels[2], channels[3])
        self.down3 = DownBlock(channels[3], channels[4])
        self.down4 = DownBlock(channels[4], channels[4])
        self.up1 = UpBlock(channels[5], channels[3])
        self.up2 = UpBlock(channels[4], channels[2])
        self.up3 = UpBlock(channels[3], channels[1])
        self.up4 = UpBlock(channels[2], channels[1])
        self.outc = OutConvBlock(channels[1], channels[6])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x.permute(0, 2, 1)
        return x


