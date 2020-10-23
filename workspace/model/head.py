import torch.nn as nn
from torch import cat
from . backbone import Darknet
from . util import Convolutional, Upsample


class YOLOv3(nn.Module):
    def __init__(self, filters):
        super(YOLOv3, self).__init__()
        self.darknet = Darknet(filters)
        self.upsample1 = nn.ModuleList([Convolutional(512, 128, 1),
                                        Upsample()])
        self.upsample2 = nn.ModuleList([Convolutional(1024, 256, 1),
                                        Upsample()])
        self.scale1 = Convolutional(256, 255, 1)
        self.scale2 = Convolutional(512, 255, 1)
        self.scale3 = Convolutional(1024, 255, 1)
        self.convs1 = nn.ModuleList([Convolutional(384, 128, 1),
                                     Convolutional(128, 256, 3),
                                     Convolutional(256, 128, 1),
                                     Convolutional(128, 256, 3),
                                     Convolutional(256, 128, 1),
                                     Convolutional(128, 256, 3)])
        self.convs2 = nn.ModuleList([Convolutional(768, 256, 1),
                                     Convolutional(256, 512, 3),
                                     Convolutional(512, 256, 1),
                                     Convolutional(256, 512, 3),
                                     Convolutional(512, 256, 1),
                                     Convolutional(256, 512, 3)])
        self.convs3 = nn.ModuleList([Convolutional(1024, 512, 1),
                                     Convolutional(512, 1024, 3),
                                     Convolutional(1024, 512, 1),
                                     Convolutional(512, 1024, 3),
                                     Convolutional(1024, 512, 1),
                                     Convolutional(512, 1024, 3)])

    def forward(self, x):
        scale1, scale2, x = self.darknet(x)

        # Scale3
        for layer in self.convs3:
            x = layer(x)

        scale3 = self.scale3(x)

        # Scale2
        for layer in self.upsample2:
            x = layer(x)

        x = cat((x, scale2), dim=1)
        for layer in self.convs2:
            x = layer(x)

        scale2 = self.scale2(x)

        # Scale1
        for layer in self.upsample1:
            x = layer(x)

        x = cat((x, scale1), dim=1)
        for layer in self.convs1:
            x = layer(x)

        scale1 = self.scale1(x)

        return scale1, scale2, scale3
