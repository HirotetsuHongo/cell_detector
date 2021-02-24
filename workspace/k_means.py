import sys
import os
import numpy as np
import csv
import config as cfg


def load_box(path, width, height):
    boxes = []
    with open(path, 'r') as f:
        rows = csv.reader(f, delimiter=' ')
        for row in rows:
            w = np.float(row[3])
            h = np.float(row[4])
            boxes.append(np.array([[w * width, h * height]]))
    boxes = np.concatenate(boxes, 0)
    return boxes


def k_means(k, boxes):
    for i in range(len(boxes)):
        print(boxes[i])
    n = len(boxes)
    classes = np.random.rand(n) * k // 1
    centers = np.zeros((k, 2))
    for i in range(k):
        centers[i] = np.mean(boxes[classes == i], 0)
    distances = np.zeros((n, k))
    for i in range(k):
        distances[:, i] = np.mean(np.square(boxes - centers[i]), 1)

    while np.count_nonzero(np.argmin(distances, 1) == classes) < n:
        classes = np.argmin(distances, 1)
        for i in range(k):
            if boxes[classes == i].shape[0] != 0:
                centers[i] = np.mean(boxes[classes == i], 0)
        for i in range(k):
            distances[:, i] = np.mean(np.square(boxes - centers[i]), 1)

    order = np.argsort(centers[:, 0] * centers[:, 1])

    return centers[order]


def main():
    # assert standard input arguments
    if len(sys.argv) != 3:
        print('usage: {} k bboxes_dir'.format(sys.argv[0]))
        return

    # set arguments
    k = int(sys.argv[1])
    bboxes_dir = sys.argv[2]

    # load config
    width = cfg.config['width']
    height = cfg.config['height']

    # load boxes
    boxes = [load_box(os.path.join(bboxes_dir, path), width, height)
             for path in os.listdir(bboxes_dir)]
    boxes = np.concatenate(boxes, 0)

    # k-means
    anchor_boxes = k_means(k, boxes)

    for anchor_box in anchor_boxes:
        print(anchor_box)


if __name__ == '__main__':
    main()
