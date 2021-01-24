import sys
import cv2
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Usage: %s input_image output_image" % (sys.argv[0]))
        sys.exit()

    img_path = sys.argv[1]
    output_path = sys.argv[2]
    img = cv2.imread(img_path, -1)
    img = img * (255 / img.max())
    gamma = 0.66
    img = 255 * np.power(img / 255, 1 / gamma)

    cv2.imwrite(output_path, img.astype(np.uint8))
    print("{} is converted into {}.".format(img_path, output_path))


if __name__ == '__main__':
    main()
