import torch
import torch.nn as nn
import torch.nn.functiona as F


def transform(detection, input_size, anchors):
    batch_size = detection.size(0)
    bbox_attrs = detection.size(1) / 3
    height = detection.size(2)
    width = detection.size(3)
    num_anchors = len(anchors)

    detection = detection.view(batch_size, bbox_attrs*num_anchors)
