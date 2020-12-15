import model
import config as cfg
import preprocess as pre
import postprocess as post

import time


def main():
    # constants
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
    net = net.train(False)

    # calculate loss for each weights
    for weight_path in weight_paths:
        loss = 0.0
        t0 = time.time()

        for i in range(num_images):
            # load images and targets
            image = pre.load_image(image_paths[i],
                                   height,
                                   width,
                                   cuda)
            image = image.unsqueeze(0)
            target = pre.load_bbox(target_paths[i],
                                   num_classes,
                                   height,
                                   width,
                                   cuda)

            # predictions
            predictions = net(image)

            # calculate loss
            loss_i = post.calculate_loss(predictions,
                                         target,
                                         anchors,
                                         height,
                                         width,
                                         cuda)
            loss += loss_i

        loss /= num_images
        elapsed_time = time.time() - t0

        print("Weight: {}, Time: {:.3f}, Loss: {:.3f}"
              .format(weight_path, elapsed_time, loss))


if __name__ == '__main__':
    main()
