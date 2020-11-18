import torch
import torch.nn as nn
from torch import cat
from . backbone import Darknet
from . util import Convolutional, Upsample


class YOLOv3(nn.Module):
    def __init__(self, num_filters, num_classes, confidency):
        super(YOLOv3, self).__init__()
        self.darknet = Darknet(num_filters)
        self.upsample1 = nn.ModuleList([Convolutional(512, 128, 1),
                                        Upsample()])
        self.upsample2 = nn.ModuleList([Convolutional(1024, 256, 1),
                                        Upsample()])
        self.scale1 = Convolutional(256, 3 * (4 + 1 + num_classes), 1)
        self.scale2 = Convolutional(512, 3 * (4 + 1 + num_classes), 1)
        self.scale3 = Convolutional(1024, 3 * (4 + 1 + num_classes), 1)
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
        self.confidency = confidency

    def forward(self, x):
        def transform(scales):
            scales = [scale.transpose(1, 3)
                           .reshape(scale.size(0),
                                    scale.size(3) * scale.size(2) * 3,
                                    scale.size(1) // 3)
                      for scale in scales]
            return torch.cat(scales, 1)

        def extract_bboxes(x, confidency):
            x = filter_feature(lambda x: x[:, :, 4] > confidency, x)
            return x

        def filter_feature(fn, x):
            x = x * fn(x).float().unsqueeze(2)
            return x

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

        # transform and concat scales
        x = transform((scale1, scale2, scale3))

        # extract bboxes
        x = extract_bboxes(x, self.confidency)

        return x
