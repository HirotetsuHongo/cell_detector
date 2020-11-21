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
        t0 = time.time()
        scale1, scale2, x = self.darknet(x)
        t = time.time()
        print('Darknet: {:.5f}'.format(t-t0))

        t0 = time.time()
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
        t = time.time()
        print('YOLOv3 expect Darknet: {:.5f}'.format(t-t0))

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
        t1 = time.time()
        print('YOLOv3: {:.5f} sec'.format(t1-t0))

        # transform feature maps
        t0 = time.time()
        for xi in x:
            # define variables
            batch_size = xi[0]
            height = xi[1]
            width = xi[2]
            num_anchors = 3
            bbox_size = xi[3] // num_anchors

            # reshape the feature map
            t0 = time.time()
            xi = xi.transpose(1, 3)
            xi = xi.reshape(batch_size, height*width*num_anchors, bbox_size)
            t1 = time.time()
            print('reshape: {:.5f} sec'.format(t1-t0))

            # apply sigmoid to the confidency
            # and classes probability distribution
            t0 = time.time()
            xi[:, :, 4:] = torch.sigmoid(xi[:, :, 4:])
            t1 = time.time()
            print('sigmoid confidency and probability: {:.5f} sec'
                  .format(t1-t0))

            # apply sigmoid to the tx and ty
            t0 = time.time()
            xi[:, :, 0] = torch.sigmoid(xi[:, :, 0])
            xi[:, :, 1] = torch.sigmoid(xi[:, :, 1])
            t1 = time.time()
            print('sigmoid tx and ty: {:.5f} sec'.format(t1-t0))

        # t1 = time.time()
        # print('transform: {:.5f} sec'.format(t1-t0))
        # # concatnate feature maps into single feature map
        # t0 = time.time()
        # x = torch.cat(x, 1)
        # t1 = time.time()
        # print('concatenate: {:.5f} sec'.format(t1-t0))

        # # add cx and cy to tx and ty
        # t0 = time.time()
        # cx = torch.arange(x.size(2))
        # cy = torch.arange(self.input_height)
        # if self.CUDA:
        #     cx = cx.cuda()
        #     cy = cy.cuda()
        # cx, cy = torch.meshgrid(cx, cy)
        # cx = cx.reshape(-1, 1)
        # cy = cy.reshape(-1, 1)
        # cxy = torch.cat((cx, cy), 1).repeat(1, 3).reshape(-1, 2).unsqueeze(0)
        # x = x, cxy
        # t1 = time.time()
        # print('add cx and cy: {:.5f} sec'.format(t1-t0))

        return x


x = torch.randn(6, 1, 416, 416).cuda()
f = Net(416, 416, 1, 4, 0.5).cuda()
