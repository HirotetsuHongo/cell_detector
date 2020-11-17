import torch.nn as nn


class Convolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, bn=True, activate=True):
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=int(((kernel_size-1) / 2)))
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if activate:
            self.activate = nn.LeakyReLU()
        else:
            self.activate = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class Residual(nn.Module):
    def __init__(self, channels1, channels2):
        super(Residual, self).__init__()
        self.layer1 = Convolutional(channels1, channels2, 1)
        self.layer3 = Convolutional(channels2, channels1, 3)

    def forward(self, x):
        shortcut = x
        x = self.layer1(x)
        x = self.layer3(x)
        x = x + shortcut
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.upsample(x)
        return x
