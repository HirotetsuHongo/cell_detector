import numpy as np
import cv2
import torch


def load_images(paths):
    images = [cv2.imread(path, -1).astype(np.float) for path in paths]
    return images


def preprocess(images, height, width, CUDA=True):
    # resize
    images = np.array([cv2.resize(image, (height, width)) for image in images])

    # convert into tensor
    images = torch.from_numpy(images)
    if CUDA:
        images = images.cuda()

    if len(images.size()) == 3:
        images = images.unsqueeze(1)
    elif len(images.size()) != 4:
        raise ValueError('Unexcected torch.size.')

    return images
