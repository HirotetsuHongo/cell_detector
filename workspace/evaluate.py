import config as cfg
import preprocess as pre
import postprocess as post

import torch

import sys
import os


def main():
    # input assertion
    if len(sys.argv) != 3:
        print('usage: {} predictions_dir targets_dir'.format(sys.argv[0]))
        return

    # constants
    num_classes = len(cfg.config['classes'])
    height = cfg.config['height']
    width = cfg.config['width']
    cuda = cfg.config['cuda']

    # inputs
    prediction_paths = sorted(os.listdir(sys.argv[1]))
    target_paths = sorted(os.listdir(sys.argv[2]))

    mAP = 0.0
    for (prediction_path, target_path) in zip(prediction_paths, target_paths):
        print('Calculating mAP of {} and {}.'
              .format(prediction_path, target_path))
        pbbox = pre.load_bbox(prediction_path, num_classes, height, width, cuda)
        tbbox = pre.load_bbox(target_path, num_classes, height, width, cuda)

    return
