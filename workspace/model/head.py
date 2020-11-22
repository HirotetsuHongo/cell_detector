import time
import torch
import torch.nn as nn
from . backbone import Darknet
from . util import Convolutional, Upsample


class YOLOv3(nn.Module):
    def __init__(self, num_filters, num_classes):
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

    def forward(self, x):
        scale1, scale2, x = self.darknet(x)

        # Scale3
        for layer in self.convs3:
            x = layer(x)

        scale3 = self.scale3(x)

        # Scale2
        for layer in self.upsample2:
            x = layer(x)

        x = torch.cat((x, scale2), 1)
        for layer in self.convs2:
            x = layer(x)

        scale2 = self.scale2(x)

        # Scale1
        for layer in self.upsample1:
            x = layer(x)

        x = torch.cat((x, scale1), 1)
        for layer in self.convs1:
            x = layer(x)

        scale1 = self.scale1(x)

        return scale1, scale2, scale3


class Net(nn.Module):
    def __init__(self, input_height, input_width, num_filters,
                 num_classes, confidency, CUDA=True):
        super(Net, self).__init__()
        self.yolov3 = YOLOv3(num_filters, num_classes)
        self.input_height = input_height
        self.input_width = input_width
        self.confidency = confidency
        self.CUDA = CUDA

    def forward(self, x):
        # apply YOLOv3
        t0 = time.time()
        x = self.yolov3(x)
        x = list(x)
        t1 = time.time()
        print('YOLOv3: {:.5f} ms'.format((t1-t0)*1000))

        # transform feature maps
        t0 = time.time()
        for i in range(len(x)):
            # define variables
            batch_size = x[i].size(0)
            height = x[i].size(2)
            width = x[i].size(3)
            num_anchors = 3
            bbox_size = x[i].size(1) // num_anchors

            # reshape the feature map
            x[i] = x[i].transpose(1, 3)
            x[i] = x[i].reshape(batch_size,
                                height*width*num_anchors,
                                bbox_size)

            # apply sigmoid to the confidency
            # and classes probability distribution
            x[i][:, :, 4:] = torch.sigmoid(x[i][:, :, 4:])

            # apply sigmoid to the tx and ty
            x[i][:, :, 0] = torch.sigmoid(x[i][:, :, 0])
            x[i][:, :, 1] = torch.sigmoid(x[i][:, :, 1])

            # add cx and cy to tx and ty
            tt0 = time.time()
            cx = torch.arange(width)
            cy = torch.arange(height)
            if self.CUDA:
                cx = cx.cuda()
                cy = cy.cuda()
            tt1 = time.time()
            print('arange: {:.5f} ms'.format((tt1-tt0)*1000))
            cx, cy = torch.meshgrid(cx, cy)
            cx = cx.unsqueeze(2)
            cy = cy.unsqueeze(2)
            cxy = torch.cat((cx, cy), 2)
            cxy = cxy.repeat(1, 1, num_anchors)
            cxy = cxy.reshape(-1, 2)
            x[i][:, :, :2] += cxy

            # normalize
            x[i][:, :, :4] *= (self.input_height // height)

        t1 = time.time()
        print('transform: {:.5f} ms'.format((t1-t0)*1000))

        # concatnate feature maps into single feature map
        t0 = time.time()
        x = torch.cat(x, 1)
        t1 = time.time()
        print('concatenate: {:.5f} ms'.format((t1-t0)*1000))

        return x
