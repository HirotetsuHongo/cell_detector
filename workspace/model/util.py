import numpy as np
import torch
import torch.nn as nn
from config import config


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


def transform_feature_map(feature_map, input_size, anchor1, anchor2, anchor3):
    batch_size, depth, size, _ = feature_map.size()
    num_anchors = 3
    depth = depth // num_anchors
    num_classes = depth - 5
    stride = input_size // size

    # set anchors
    anchors = [(a[0]/stride, a[1]/stride) for a in [anchor1, anchor2, anchor3]]

    # reshape feature_map
    feature_map = feature_map.view(batch_size, depth*num_anchors, size*size)
    feature_map = feature_map.transpose(1, 2).contiguous()
    feature_map = feature_map.view(batch_size, size*size*num_anchors, depth)

    # apply sigmoid to X, Y and confidency.
    feature_map[:, :, 0] = torch.sigmoid(feature_map[:, :, 0])  # X
    feature_map[:, :, 1] = torch.sigmoid(feature_map[:, :, 1])  # Y
    feature_map[:, :, 4] = torch.sigmoid(feature_map[:, :, 4])  # confidency

    # add the center offsets
    grid = np.arange(size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if config['CUDA'] == 'Enable':
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    xy_offset = torch.cat((x_offset, y_offset), 1) \
                     .repeat(1, num_anchors) \
                     .view(-1, 2) \
                     .unsqueeze(0)
    feature_map[:, :, :2] += xy_offset

    # log spase transform height and the width
    anchors = torch.FloatTensor(anchors)

    if config['CUDA'] == 'Enable':
        anchors = anchors.cuda()

    anchors = anchors.repeat(size*size, 1).unsqueeze(0)
    feature_map[:, :, 2:4] = torch.exp(feature_map[:, :, 2:4])*anchors

    # apply sigmoid to the class distribution
    feature_map[:, :, 5:5+num_classes] = \
        torch.sigmoid((feature_map[:, :, 5:5+num_classes]))

    # resize bbox attributes.
    feature_map[:, :, :4] *= stride

    return feature_map
