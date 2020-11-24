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
    def __init__(self, input_height, input_width, num_filters, num_classes,
                 anchors, confidency_threshold=0.5, CUDA=True):
        super(Net, self).__init__()
        self.yolov3 = YOLOv3(num_filters, num_classes)
        self.input_height = input_height
        self.input_width = input_width
        self.anchors = torch.Tensor(anchors)
        self.confidency = confidency_threshold
        self.CUDA = CUDA
        if self.CUDA:
            self.anchors = self.anchors.cuda()

    def forward(self, x):
        # apply YOLOv3
        x = self.yolov3(x)
        x = list(x)

        # transform feature maps
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
            cx = torch.arange(width)
            cy = torch.arange(height)
            if self.CUDA:
                cx = cx.cuda()
                cy = cy.cuda()
            cx, cy = torch.meshgrid(cx, cy)
            cx = cx.unsqueeze(2)
            cy = cy.unsqueeze(2)
            cxy = torch.cat((cx, cy), 2)
            cxy = cxy.repeat(1, 1, num_anchors)
            cxy = cxy.reshape(-1, 2)
            x[i][:, :, :2] += cxy

            # log space transform height and width using anchors
            anchors = self.anchors[i].repeat(height*width, 1).unsqueeze(0)
            x[i][:, :, 2:4] = torch.exp(x[i][:, :, 2:4])
            x[i][:, :, 2:4] *= anchors

            # normalize
            x[i][:, :, 0] *= (self.input_width // width)
            x[i][:, :, 1] *= (self.input_height // height)
            x[i][:, :, 2] *= (self.input_width // width)
            x[i][:, :, 3] *= (self.input_height // height)

        # concatnate feature maps into single feature map
        x = torch.cat(x, 1)

        # split by batch size
        x = [x[i] for i in range(x.size(0))]

        t0 = time.time()
        for i in range(len(x)):
            # threshold by confidency
            mask = (x[i][:, 4] > self.confidency).unsqueeze(1)
            x[i] = torch.masked_select(x[i], mask).reshape(-1, 9)

            # convert bbox part
            # from (center x, center y, width, height)
            # to (xmin, ymin, xmax, ymax)
            bbox = torch.zeros(x[i].size(0), 4)
            bbox[:, 0] = (x[i][:, 0] - x[i][:, 2]/2)
            bbox[:, 1] = (x[i][:, 0] + x[i][:, 2]/2)
            bbox[:, 2] = (x[i][:, 1] - x[i][:, 3]/2)
            bbox[:, 3] = (x[i][:, 1] + x[i][:, 3]/2)
            x[i][:, :4] = bbox

            # select class
            max_class, max_class_id = torch.max(x[i][:, 5:], 1)
            max_class = max_class.unsqueeze(1)
            max_class_id = max_class_id.unsqueeze(1)
            x[i] = torch.cat((x[i][:, :5], max_class, max_class_id), 1)

        t1 = time.time()
        print("Elapsed time of precesses per a picture: {:.3f} ms"
              .format((t1-t0)*1000))

        # concatnate feature maps into single feature map
        # x = torch.cat(x, 0)

        return x


x = torch.randn(6, 1, 416, 416).cuda()

anchors = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]

f = Net(416, 416, 1, 4, anchors).cuda()
