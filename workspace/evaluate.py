import config as cfg
import preprocess as pre
import postprocess as post

import torch

import sys
import os


def main():
    # input assertion
    if len(sys.argv) != 3:
        print('usage: {} prediction_path target_path'.format(sys.argv[0]))
        return

    # inputs
    prediction_path = os.listdir(sys.argv[1])
    targets_path = sys.argv[2]

    # constants
    prediction_paths = os.listdir(prediction_path)
    target_paths = os.listdir(targets_path)

    return
