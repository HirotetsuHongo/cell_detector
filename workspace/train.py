import model
import config as cfg
import preprocess as pre
import postprocess as post

import os
import datetime
import torch
import torch.optim as optim


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
    cuda = cfg.config['CUDA']
    images_path = cfg.config['path']['image']
    targets_path = cfg.config['path']['train']
    weight_path = cfg.config['path']['weight']
    initial_weight_path = cfg.config['path']['initial_weight']
    image_paths = load_image_paths(images_path, targets_path)
    target_paths = load_bbox_paths(targets_path)
    num_images = len(image_paths)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d_%H:%M:%S')

    # network
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()
    net = net.train(True)
    if initial_weight_path:
        net.load_state_dict(torch.load(initial_weight_path))

    # optimizer
    optimizer = optim.Adam(net.parameters())

    # train
    for epoch in range(num_epochs):
        losses = []

        for i in range((num_images // batch_size) + 1):
            start = batch_size * i
            end = min(start + batch_size, num_images)
            if start >= end:
                break
            images = load_images(image_paths[start:end], height, width, cuda)
            targets = load_targets(target_paths[start:end],
                                   num_classes,
                                   height,
                                   width,
                                   cuda)
            loss = train(net, optimizer,
                         images, targets,
                         anchors, height, width, cuda)
            print("Epoch: {}, Batch: {}, Loss: {:.3f}".format(epoch, i, loss))
            losses.append(loss)

        # save weight
        loss = sum(losses) / len(losses)
        filename = "{}_{}_{}.pt".format(now, epoch, loss)
        full_filename = os.path.join(weight_path, filename)
        torch.save(net.state_dict(), full_filename)
        print("Saved {}.".format(full_filename))


def train(net, optimizer, images, targets, anchors, height, width, cuda):
    assert len(images) == len(targets)
    images = [image.unsqueeze(0) for image in images]
    images = torch.cat(images, 0)
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


def load_image_paths(images_path, bboxes_path):
    basenames = os.listdir(bboxes_path)
    basenames = [os.path.splitext(name)[0] for name in basenames]
    paths = os.listdir(images_path)
    paths = [path for path in paths if os.path.splitext(path)[0] in basenames]
    paths = [os.path.join(images_path, path) for path in paths]
    paths = sorted(paths)
    return paths


def load_bbox_paths(bboxes_path):
    paths = os.listdir(bboxes_path)
    paths = [os.path.join(bboxes_path, path) for path in paths]
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
