import csv
import numpy as np
import cv2
import torch


def load_image(path, height, width, cuda):
    image = cv2.imread(path, -1)
    image = image.astype(np.float)
    image = np.resize(image, (height, width))
    image = torch.from_numpy(image)
    if cuda:
        image = image.cuda()

    if len(image.size()) == 2:
        image = image.unsqueeze(0)
    elif len(image.size()) != 3:
        raise Exception("Unexpected image size")

    return image


def load_bbox(path, height, width, cuda):
    bboxes = []
    with open(path, 'r') as f:
        rows = csv.reader(f, delimiter=' ')
        for row in rows:
            [cid, x, y, w, h] = [np.float(e) for e in row]
            bbox = np.array([[x * width,
                              y * height,
                              w * width,
                              h * height,
                              1,
                              cid]])
            bboxes.append(bbox)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    if cuda:
        bboxes = bboxes.cuda()

    return bboxes
