import os
import detect
import time


def main():
    images_dir = 'images'
    results_dir = 'results'
    images = os.listdir(images_dir)
    names = [os.path.splitext(img)[0] for img in images]
    images = [os.path.join(images_dir, img) for img in images]
    results = [os.path.join(results_dir, name + '.txt') for name in names]

    t0 = time.time()
    for image, result in zip(images, results):
        detect.detect(image, result)
    print('Elapsed Time: {}s'.format(time.time() - t0))


if __name__ == '__main__':
    main()
