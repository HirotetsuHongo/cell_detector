import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


def extract_bboxes(feature_maps):
    feature_map = torch.cat([transform_feature_map(fm)
                             for fm in feature_maps],
                            1)
    return feature_map


