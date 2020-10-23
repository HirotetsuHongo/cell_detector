import torch.nn as nn
from . util import Convolutional, Residual


class Darknet(nn.Module):
    def __init__(self, filters):
        super(Darknet, self).__init__()
        self.layers1 = nn.ModuleList([Convolutional(filters, 32, 3),
                                      Convolutional(32, 64, 3, stride=2),
                                      Residual(64, 32),
                                      Convolutional(64, 128, 3, stride=2),
                                      Residual(128, 64),
                                      Residual(128, 64),
                                      Convolutional(128, 256, 3, stride=2),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128),
                                      Residual(256, 128)])
        self.layers2 = nn.ModuleList([Convolutional(256, 512, 3, stride=2),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256),
                                      Residual(512, 256)])
        self.layers3 = nn.ModuleList([Convolutional(512, 1024, 3, stride=2),
                                      Residual(1024, 512),
                                      Residual(1024, 512),
                                      Residual(1024, 512),
                                      Residual(1024, 512)])

    def forward(self, x):
        for layer in self.layers1:
            x = layer(x)

        route1 = x
        for layer in self.layers2:
            x = layer(x)

        route2 = x
        for layer in self.layers3:
            x = layer(x)

        return route1, route2, x
