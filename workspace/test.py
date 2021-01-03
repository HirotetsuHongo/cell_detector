import model
import config as cfg
import preprocess as pre
import postprocess as post

import torch

import sys
import time


def main():
    # input assertion
    if len(sys.argv) != 2 or (sys.argv[1] != 'loss' and sys.argv[1] != 'AP'):
        print('Usage: {} MODE'.format(sys.argv[0]))
        print('  MODE: loss | AP')
        return

    # constants
    mode = sys.argv[1]
    num_channels = cfg.config['num_channels']
    classes = cfg.config['classes']
    num_classes = len(classes)
    height = cfg.config['height']
    width = cfg.config['width']
    anchors = cfg.config['anchors']
    cuda = cfg.config['CUDA']
    target_dir = cfg.config['path']['test']
    image_dir = cfg.config['path']['image']
    weight_dir = cfg.config['path']['weight_test']
    image_paths = pre.load_image_paths(image_dir, target_dir)
    target_paths = pre.load_dir_paths(target_dir)
    weight_paths = pre.load_dir_paths(weight_dir)
    num_images = len(image_paths)

    # network
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()

    # calculate loss or mAP for each weights
    for weight_path in weight_paths:
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        output = 0.0
        t0 = time.time()

        for i in range(num_images):
            # load images and targets
            image = pre.load_images(image_paths[i:i+1],
                                    height,
                                    width,
                                    cuda)
            target = pre.load_targets(target_paths[i:i+1],
                                      num_classes,
                                      height,
                                      width,
                                      cuda)

            # generate prediction
            prediction = net(image)

            # calculate loss or 
            if mode == 'loss':
                output += post.calculate_loss(prediction,
                                              target,
                                              anchors,
                                              height,
                                              width,
                                              cuda)
            elif mode == 'AP':
                output += post.calculate_AP(prediction,
                                            target,
                                            anchors,
                                            height,
                                            width,
                                            cuda)

        output /= num_images
        elapsed_time = time.time() - t0

        if mode == 'loss':
            print("Weight: {}, Elapsed Time: {:.2f}s, Loss: {:.2f}"
                  .format(weight_path, elapsed_time, output))
        elif mode == 'AP':
            print("Weight: {}, Elapsed Time: {:.2f}s, mAP: {:.2f}"
                  .format(weight_path, elapsed_time, output))


if __name__ == '__main__':
    main()
