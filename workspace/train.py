import model
import config as cfg
import preprocess as pre
import postprocess as post

import torch
import torch.optim as optim

import os
import datetime
import time


def main():
    # constants
    batch_size = cfg.config['batch_size']
    num_channels = cfg.config['num_channels']
    classes = cfg.config['classes']
    num_classes = len(classes)
    height = cfg.config['height']
    width = cfg.config['width']
    anchors = cfg.config['anchors']
    num_epochs = cfg.config['num_epochs']
    learning_rate = cfg.config['learning_rate']
    cuda = cfg.config['CUDA']
    image_dir = cfg.config['path']['image']
    target_dir = cfg.config['path']['train']
    weight_path = cfg.config['path']['weight']
    initial_weight_path = cfg.config['path']['initial_weight']
    image_paths = load_image_paths(image_dir, target_dir)
    target_paths = load_bbox_paths(target_dir)
    num_images = len(image_paths)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d_%H-%M-%S')
    weight_file = False

    # network
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()
    net = net.train(True)
    if initial_weight_path:
        net.load_state_dict(torch.load(initial_weight_path))
        print('Load from {}.'.format(initial_weight_path))

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        losses = []
        t0 = time.time()

        for i in range((num_images // batch_size) + 1):
            # get indices
            start = batch_size * i
            end = min(start + batch_size, num_images)
            if start >= end:
                break

            # load images and targets
            images = load_images(image_paths[start:end], height, width, cuda)
            targets = load_targets(target_paths[start:end],
                                   num_classes,
                                   height,
                                   width,
                                   cuda)

            # train
            loss = train(net, optimizer,
                         images, targets,
                         anchors, height, width, cuda)

            # NaN
            if torch.isnan(loss):
                print("NaN is occured.")
                if weight_file:
                    net.load_state_dict(torch.load(weight_file))
                    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                    print("Reset weight to {}.".format(weight_file))
                    continue
                else:
                    print("Previous weight does not exist.")
                    break
            losses.append(loss)

        # NaN
        if torch.isnan(loss):
            break

        # calculate average of loss
        loss = sum(losses) / len(losses)

        # time elapse
        elapsed_time = time.time() - t0
        print("Epoch: {}, Time: {:.3f}s, Loss: {:.3f}"
              .format(epoch, elapsed_time, loss))

        # save weight
        text = "{}_{:0>4}_{:.3f}.pt".format(now, epoch, loss)
        weight_file = os.path.join(weight_path, text)
        torch.save(net.state_dict(), weight_file)
        print("Saved {}.".format(weight_file))


def train(net, optimizer, images, targets, anchors, height, width, cuda):
    assert len(images) == len(targets)
    images = [image.unsqueeze(0) for image in images]
    images = torch.cat(images, 0)
    optimizer.zero_grad()
    predictions = net(images)
    loss = post.calculate_loss(predictions,
                               targets,
                               anchors,
                               height,
                               width,
                               cuda)
    loss.backward()
    optimizer.step()
    return loss


def load_image_paths(image_dir, bbox_dir):
    basenames = os.listdir(bbox_dir)
    basenames = [os.path.splitext(name)[0] for name in basenames]
    paths = os.listdir(image_dir)
    paths = [path for path in paths if os.path.splitext(path)[0] in basenames]
    paths = [os.path.join(image_dir, path) for path in paths]
    paths = sorted(paths)
    return paths


def load_bbox_paths(bbox_dir):
    paths = os.listdir(bbox_dir)
    paths = [os.path.join(bbox_dir, path) for path in paths]
    paths = sorted(paths)
    return paths


def load_images(image_paths, height, width, cuda):
    images = [pre.load_image(path, height, width, cuda)
              for path in image_paths]
    return images


def load_targets(bbox_paths, num_classes, height, width, cuda):
    targets = [pre.load_bbox(path, num_classes, height, width, cuda)
               for path in bbox_paths]
    return targets


if __name__ == '__main__':
    main()
