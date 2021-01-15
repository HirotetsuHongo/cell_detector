import torch
import torch.nn as nn
from . backbone import Darknet
from . util import Convolutional, Upsample


class YOLOv3(nn.Module):
    def __init__(self, num_channels, num_classes, num_anchors):
        super(YOLOv3, self).__init__()
        self.darknet = Darknet(num_channels)
        self.upsample1 = nn.ModuleList([Convolutional(512, 128, 1),
                                        Upsample()])
        self.upsample2 = nn.ModuleList([Convolutional(1024, 256, 1),
                                        Upsample()])
        self.scale1 = Convolutional(256,
                                    num_anchors * (4 + 1 + num_classes),
                                    1)
        self.scale2 = Convolutional(512,
                                    num_anchors * (4 + 1 + num_classes),
                                    1)
        self.scale3 = Convolutional(1024,
                                    num_anchors * (4 + 1 + num_classes),
                                    1)
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
        prediction1, prediction2, x = self.darknet(x)

        # Scale3
        for layer in self.convs3:
            x = layer(x)

        prediction3 = self.scale3(x)

        # Scale2
        for layer in self.upsample2:
            x = layer(x)

        x = torch.cat((x, prediction2), 1)
        for layer in self.convs2:
            x = layer(x)

        prediction2 = self.scale2(x)

        # Scale1
        for layer in self.upsample1:
            x = layer(x)

        x = torch.cat((x, prediction1), 1)
        for layer in self.convs1:
            x = layer(x)

        prediction1 = self.scale1(x)

        return prediction1, prediction2, prediction3
