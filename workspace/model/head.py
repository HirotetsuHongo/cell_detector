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


def postprocess(scales, anchors, input_height, input_width,
                objectness_threshold, iou_threshold, CUDA=True):
    def reshape(scale, batch_size, height, width, num_anchors, bbox_size):
        scale = scale.transpose(1, 3)
        scale = scale.reshape(batch_size,
                              height * width * num_anchors,
                              bbox_size)
        return scale

    def sigmoid_objectness(scale):
        scale[:, :, 4] = torch.sigmoid(scale[:, :, 4])
        return scale

    def adjust_coordinate(scale, height, width, stride, num_anchors, anchors):
        # sigmoid x and y
        scale[:, :, 0:2] = torch.sigmoid(scale[:, :, 0:2])

        # add center coordinate of bbox
        grid_x = torch.arange(width)
        grid_y = torch.arange(height)
        if CUDA:
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
        center_x, center_y = torch.meshgrid(grid_x, grid_y)
        center_x = center_x.unsqueeze(-1)
        center_y = center_y.unsqueeze(-1)
        center = torch.cat((center_x, center_y), -1)
        center = center.repeat(1, num_anchors, 1).reshape(-1, 2)
        scale[:, :, 0:2] += center

        # normalize x and y
        scale[:, :, 0] *= stride[0]
        scale[:, :, 1] *= stride[1]

        # log scale transform w, h
        scale[:, :, 2:4] = torch.exp(scale[:, :, 2:4])
        # use anchors
        anchors = [[a[0] / stride[0], a[1] / stride[1]] for a in anchors]
        anchors = torch.tensor(anchors)
        if CUDA:
            anchors = anchors.cuda()
        anchors = anchors.repeat(width * height, 1)
        scale[:, :, 2:4] *= anchors

        # normalize w and h
        scale[:, :, 2] *= stride[0]
        scale[:, :, 3] *= stride[1]

        return scale

    def low_objectness_suppress(image, objectness_threshold):
        mask = image[:, 4] >= objectness_threshold
        return image[mask, :]

    def select_class(image):
        class_id = torch.max(image[:, 5:], 1)[1]
        class_id = class_id.unsqueeze(1)
        image = torch.cat((image[:, :5], class_id), 1)
        return image

    def non_maximum_suppress(image, iou_threshold):
        iou_mask = bbox_iou(image.unsqueeze(-1), image) > iou_threshold
        conf_mask = image[:, 4].unsqueeze(-1) >= image[:, 4]
        id_mask = image[:, 5].unsqueeze(-1) == image[:, 5]
        mask = torch.logical_and(iou_mask, conf_mask)
        mask = torch.logical_and(mask, id_mask)
        mask = torch.any(mask, 1)
        return image

    def bbox_iou(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
        x2, y2, w2, h2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

        area1 = (w1 + 1) * (h1 + 1)
        area2 = (w2 + 1) * (h2 + 1)
        inter_w = torch.clamp(((w1 + w2) / 2) - torch.abs(x1 - x2), min=0)
        inter_h = torch.clamp(((h1 + h2) / 2) - torch.abs(y1 - y2), min=0)
        inter_area = (inter_w + 1) * (inter_h + 1)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area

        return iou

    scales = list(scales)
    for i in range(len(scales)):
        batch_size = scales[i].size(0)
        height = scales[i].size(2)
        width = scales[i].size(3)
        num_anchors = len(anchors[i])
        bbox_size = scales[i].size(1) // num_anchors
        stride = (input_width / width, input_height / height)

        scales[i] = reshape(scales[i],
                            batch_size, height, width, num_anchors, bbox_size)
        scales[i] = sigmoid_objectness(scales[i])
        scales[i] = adjust_coordinate(scales[i],
                                      height, width, stride,
                                      num_anchors, anchors[i])

    images = torch.cat(scales, 1)
    images = [images[i] for i in range(images.size(0))]

    for i in range(len(images)):
        images[i] = low_objectness_suppress(images[i], objectness_threshold)
        images[i] = select_class(images[i])
        images[i] = non_maximum_suppress(images[i], iou_threshold)

    return images
