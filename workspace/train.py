import model
import config as cfg
import preprocess as pre
import postprocess as post

import torch
import torch.optim as optim
import numpy as np

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
    num_anchors = len(anchors[0])
    num_epochs = cfg.config['num_epochs']
    learning_rate = cfg.config['learning_rate']
    weight_decay = cfg.config['weight_decay']
    cuda = cfg.config['CUDA']
    image_dir = cfg.config['path']['image']
    target_dir = cfg.config['path']['train']
    weight_dir = cfg.config['path']['weight']
    initial_weight_dir = cfg.config['path']['initial_weight']
    image_paths = pre.load_image_paths(image_dir, target_dir)
    target_paths = pre.load_dir_paths(target_dir)
    num_images = len(image_paths)
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d_%H-%M-%S')
    weight_file = False

    # network
    net = model.YOLOv3(num_channels, num_classes, num_anchors)
    if cuda:
        net = net.cuda()
    net = net.train(True)
    if initial_weight_dir:
        net.load_state_dict(torch.load(initial_weight_dir))
        print('Load initial weight from {}.'.format(initial_weight_dir))

    # optimizer
    optimizer = optim.AdamW(net.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay)

    # train
    for epoch in range(num_epochs):
        losses_coord = []
        losses_obj_cls = []
        t0 = time.time()

        for i in range((num_images // batch_size) + 1):
            # get indices
            start = batch_size * i
            end = min(start + batch_size, num_images)
            if start >= end:
                break

            # load images and targets
            images = pre.load_images(image_paths[start:end],
                                     height,
                                     width,
                                     cuda)
            targets = pre.load_targets(target_paths[start:end],
                                       num_classes,
                                       height,
                                       width,
                                       cuda)

            # train
            loss_coord, loss_obj_cls = train(net, optimizer,
                                             images, targets,
                                             anchors, height, width, cuda)

            # NaN
            if np.isnan(loss_coord) or np.isnan(loss_obj_cls):
                print("NaN is occured. Loss: {:.2f} {:.2f}"
                      .format(loss_coord, loss_obj_cls))
                if weight_file:
                    net.load_state_dict(torch.load(weight_file))
                    optimizer = optim.AdamW(net.parameters(),
                                            lr=learning_rate,
                                            weight_decay=weight_decay)
                    print("Reset weight to {}.".format(weight_file))
                    break
                else:
                    print("Previous weight does not exist.")
                    break

            losses_coord.append(loss_coord)
            losses_obj_cls.append(loss_obj_cls)

        # NaN
        if np.isnan(loss_coord) or np.isnan(loss_obj_cls):
            continue

        # calculate average of loss
        loss_coord = sum(losses_coord) / len(losses_coord)
        loss_obj_cls = sum(losses_obj_cls) / len(losses_obj_cls)

        # time elapse
        elapsed_time = time.time() - t0
        print(("Epoch: {}, Elapsed Time: {:.2f}s, " +
               "Coordinate Loss: {:.2f}, " +
               "Objectness and Class Loss: {:.2f}, " +
               "Loss: {:.2f}")
              .format(epoch, elapsed_time,
                      loss_coord, loss_obj_cls, loss_coord + loss_obj_cls))

        # save weight
        text = "{}_{:0>4}_{:.2f}.pt".format(now, epoch,
                                            loss_coord + loss_obj_cls)
        weight_file = os.path.join(weight_dir, text)
        torch.save(net.state_dict(), weight_file)
        print("Saved {}.".format(weight_file))


def train(net, optimizer, images, targets, anchors, height, width, cuda):
    # assert
    assert len(images) == len(targets)

    # main
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
    loss_coord = loss[0]
    loss_obj_cls = loss[1]
    loss = loss_coord + loss_obj_cls

    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()

    return float(loss_coord), float(loss_obj_cls)


if __name__ == '__main__':
    main()
