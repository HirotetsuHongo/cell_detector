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
    num_epochs = cfg.config['num_epochs']
    learning_rate = cfg.config['learning_rate']
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
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()
    net = net.train(True)
    if initial_weight_dir:
        net.load_state_dict(torch.load(initial_weight_dir))
        print('Load initial weight from {}.'.format(initial_weight_dir))

    # optimizer
    optimizer = optim.AdamW(net.parameters(),
                            lr=learning_rate,
                            weight_decay=0.1)

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
            loss = train(net, optimizer,
                         images, targets,
                         anchors, height, width, cuda)

            # NaN
            if torch.isnan(loss):
                print("NaN is occured.")
                if weight_file:
                    net.load_state_dict(torch.load(weight_file))
                    optimizer = optim.AdamW(net.parameters(),
                                            lr=learning_rate,
                                            weight_decay=0.1)
                    print("Reset weight to {}.".format(weight_file))
                    break
                else:
                    print("Previous weight does not exist.")
                    break

            losses.append(loss)

        # NaN
        if torch.isnan(loss):
            continue

        # calculate average of loss
        loss = sum(losses) / len(losses)

        # time elapse
        elapsed_time = time.time() - t0
        print("Epoch: {}, Elapsed Time: {:.2f}s, Loss: {:.2f}"
              .format(epoch, elapsed_time, loss))

        # save weight
        text = "{}_{:0>4}_{:.2f}.pt".format(now, epoch, loss)
        weight_file = os.path.join(weight_dir, text)
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
    if torch.isnan(loss):
        loss = torch.tensor(np.nan)
        if cuda:
            loss = loss.cuda()
    else:
        loss.backward()
        optimizer.step()

    return loss


if __name__ == '__main__':
    main()
