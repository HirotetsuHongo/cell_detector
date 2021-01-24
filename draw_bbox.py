import numpy as np
import cv2
import sys


def draw_bbox(image, bboxes, num_classes, s):
    # get width and height
    height = image.shape[0]
    width = image.shape[1]

    # make colors
    colors = np.arange(num_classes)
    colors = [(h * 359 / num_classes, s, 100) for h in colors]
    colors = [hsv2bgr(hsv) for hsv in colors]

    # draw
    for bbox in bboxes:
        c = int(bbox[0])
        x = bbox[1] * width
        y = bbox[2] * height
        w = bbox[3] * width
        h = bbox[4] * height

        color = colors[c]
        x1 = int(x - w / 2)
        y1 = int(y + h / 2)
        x2 = int(x + w / 2)
        y2 = int(y - h / 2)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

    return image


def hsv2bgr(hsv):
    h, s, v = hsv
    s = s / 100
    v = v / 100
    c = v * s
    h2 = h / 60
    x = c * (1 - abs((h2 % 2) - 1))

    r = v - c
    g = v - c
    b = v - c

    if h2 < 1:
        r += c
        g += x
    elif h2 < 2:
        r += x
        g += c
    elif h2 < 3:
        g += c
        b += x
    elif h2 < 4:
        g += x
        b += c
    elif h2 < 5:
        r += x
        b += c
    else:
        r += c
        b += x

    return (b * 255, g * 255, r * 255)


def read_table(path):
    with open(path, 'r') as f:
        result = [line.split(' ') for line in f.read().split('\n')]
    return result


def main():
    # assert standard input arguments
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print('usage: {} image bboxes output num_classes [s]'
              .format(sys.argv[0]))
        return

    # set arguments
    image_path = sys.argv[1]
    bboxes_path = sys.argv[2]
    output_path = sys.argv[3]
    num_classes = int(sys.argv[4])
    if len(sys.argv) == 6 and 0 <= int(sys.argv[5]) <= 100:
        s = int(sys.argv[5])
    else:
        s = 50

    # load image, bboxes and classes
    image = cv2.imread(image_path)
    bboxes = read_table(bboxes_path)
    while [''] in bboxes:
        bboxes.remove([''])
    bboxes = [[float(x) for x in bbox] for bbox in bboxes]

    # assert image
    assert image is not None

    # draw
    drawed_image = draw_bbox(image, bboxes, num_classes, s)

    # write image
    cv2.imwrite(output_path, drawed_image)

    print('drawed {} in {} and saved to {}.'
          .format(bboxes_path, image_path, output_path))


if __name__ == '__main__':
    main()
