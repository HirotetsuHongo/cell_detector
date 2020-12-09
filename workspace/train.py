import os
import datetime
import torch
import torch.optim as optim
import model
import config as cfg
import preprocess as pre
import postprocess as post


# Config
batch_size = cfg.config['batch_size']
num_channels = cfg.config['num_channels']
classes = cfg.config['classes']
num_classes = len(classes)
height = cfg.config['height']
width = cfg.config['width']
anchors = cfg.config['anchors']
num_epochs = cfg.config['num_epochs']
tp_iou = cfg.config['TP_IoU']
nms_iou = cfg.config['NMS_IoU']
objectness = cfg.config['objectness']
cuda = cfg.config['CUDA']
train_dir = cfg.config['path']['train']
test_dir = cfg.config['path']['test']
image_dir = cfg.config['path']['image']
weight_dir = cfg.config['path']['weight']
initial_weight_path = cfg.config['path']['initial_weight']


# Dataset
def load_images(image_dir, data_dir, cuda):
    image_files = os.listdir(image_dir)
    data_files = os.listdir(data_dir)
    data_file_names = [os.path.splitext(data_file) for data_file in data_files]

    images = [os.path.join(image_dir, image_file)
              for image_file in image_files
              if os.path.splitext(image_file) in data_file_names]
    images = [pre.load_image(image_path, height, width, cuda).unsqueeze(0)
              for image_path in images]
    images = torch.cat(images, 0)

    return images


def load_bboxes(data_dir, cuda):
    bboxes = os.listdir(data_dir)
    bboxes = [data_dir + data_file for data_file in bboxes]
    bboxes = [pre.load_bbox(path, height, width, cuda) for path in bboxes]

    return bboxes


# Train
def train_batch(net, optimizer, images, targets):
    assert images.shape[0] == len(targets)
    batch_size = images.shape[0]
    optimizer.zero_grad()
    predictions = net(images)
    predictions = post.postprocess(predictions,
                                   anchors,
                                   height,
                                   width,
                                   objectness,
                                   nms_iou,
                                   cuda)
    loss = 0
    for i in range(batch_size):
        loss += post.calculate_loss(predictions[i], targets[i], tp_iou, cuda)
    loss /= batch_size
    loss.backward()
    optimizer.step()
    return loss


def test_batch(net, images, targets):
    assert images.shape[0] == len(targets)
    batch_size = images.shape[0]
    predictions = net(images)
    predictions = post.postprocess(predictions,
                                   anchors,
                                   height,
                                   width,
                                   objectness,
                                   nms_iou,
                                   cuda)
    loss = 0
    for i in range(batch_size):
        loss += post.calculate_loss(predictions[i], targets[i], tp_iou, cuda)
    loss /= batch_size
    return loss


def train(image_dir, train_dir, test_dir, weight_dir,
          num_channels, num_classes, num_epochs,
          initial_weight_path, cuda):
    # load images and bboxes
    train_images = load_images(image_dir, train_dir, cuda)
    train_bboxes = load_bboxes(train_dir, cuda)
    test_images = load_images(image_dir, test_dir, cuda)
    test_bboxes = load_bboxes(test_dir, cuda)

    # constants
    num_train_images = train_images.shape[0]

    # net
    net = model.YOLOv3(num_channels, num_classes)
    if cuda:
        net = net.cuda()
    net = net.train(True)
    if initial_weight_path:
        net.load_state_dict(torch.load(initial_weight_path))

    # optimizer
    optimizer = optim.Adam(net.parameters())

    # training
    dt_now = datetime.date.today()
    for epoch in num_epochs:
        for i in range(num_train_images // batch_size + 1):
            start = batch_size * i
            end = min(start + 6, train_images.shape[0])
            train_batch(net,
                        optimizer,
                        train_images[start:end],
                        train_bboxes[start:end])

        loss = test_batch(net, test_images, test_bboxes)
        text = "{}_{}_{:.4g}.pt".format(dt_now, epoch, loss)
        torch.save(net.state_dict(),
                   os.path.join(weight_dir, text))
        print("Saved {}.".format(text))


def main():
    train(image_dir, train_dir, test_dir, weight_dir,
          num_channels, num_classes, num_epochs,
          initial_weight_path, cuda)


if __name__ == '__main__':
    main()
