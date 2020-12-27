import torch

import os
import csv
import numpy as np
import cv2


def load_image(path, height, width, cuda):
    image = cv2.imread(path, -1)
    image = image.astype(np.float)
    image = cv2.resize(image, (height, width))
    if image.ndim == 3:
        image = image.transpose((2, 1, 0))
    image = torch.from_numpy(image.astype(np.float32))
    if cuda:
        image = image.cuda()

    if image.ndim == 2:
        image = image.unsqueeze(0)
    elif image.ndim != 3:
        raise Exception("Unexpected image size")

    return image


def load_bbox(path, num_classes, height, width, cuda):
    bboxes = []
    with open(path, 'r') as f:
        rows = csv.reader(f, delimiter=' ')
        for row in rows:
            if len(row) == 5:
                [cid, x, y, w, h] = [np.float(e) for e in row]
                bbox = np.array([[x * width,
                                  y * height,
                                  w * width,
                                  h * height,
                                  1.0]])
            elif len(row) == 6:
                [cid, x, y, w, h, c] = [np.float(e) for e in row]
                bbox = np.array([[x * width,
                                  y * height,
                                  w * width,
                                  h * height,
                                  c]])
            else:
                assert len(row) == 5 or len(row) == 6
            classes = np.zeros((1, num_classes))
            classes[:, int(cid)] += 1.0
            bbox = np.concatenate((bbox, classes), axis=1)
            bboxes.append(bbox)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes.astype(np.float32))
    if cuda:
        bboxes = bboxes.cuda()

    return bboxes


def load_image_paths(image_dir, bbox_dir):
    basenames = os.listdir(bbox_dir)
    basenames = [os.path.splitext(name)[0] for name in basenames]
    paths = os.listdir(image_dir)
    paths = [path for path in paths if os.path.splitext(path)[0] in basenames]
    paths = [os.path.join(image_dir, path) for path in paths]
    paths = sorted(paths)
    return paths


def load_dir_paths(directory):
    paths = os.listdir(directory)
    paths = [os.path.join(directory, path) for path in paths]
    paths = sorted(paths)
    return paths


def load_images(image_paths, height, width, cuda):
    images = [load_image(path, height, width, cuda)
              for path in image_paths]
    return images


def load_targets(bbox_paths, num_classes, height, width, cuda):
    targets = [load_bbox(path, num_classes, height, width, cuda)
               for path in bbox_paths]
    return targets
