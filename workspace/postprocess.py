import torch
import torch.nn as nn
import torch.nn.functiona as F


def transform(feature_map, input_size, anchor1, anchor2, anchor3):
    batch_size, depth, size, _ = feature_map
    depth = depth / 3
    stride = input_size / size

    # set anchors
    anchors = [(a[0]/stride, a[1]/stride) for a in [anchor1, anchor2, anchor3]]

    # reshape feature_map
    feature_map = feature_map.view(batch_size, depth*3, size*size)
    feature_map = feature_map.transpose(1, 2).contiguous()
    feature_map = feature_map.view(batch_size, size*size*3, depth)

    # Sigmoid X, Y and confidency.
    feature_map[:,:,0] = torch.sigmoid(feature_map[:,:,0]) # X
    feature_map[:,:,1] = torch.sigmoid(feature_map[:,:,1]) # Y
    feature_map[:,:,4] = torch.sigmoid(feature_map[:,:,4]) # confidency
