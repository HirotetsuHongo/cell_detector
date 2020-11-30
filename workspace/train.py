import os
import torch
import torch.nn as nn
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
num_epochs = cfg.config['epochs']
tp_iou = cfg.config['TP_IoU']
nms_iou = cfg.config['NMS_IoU']
objectness = cfg.config['objectness']
cuda = cfg.config['CUDA']
train_dir = cfg.config['dir']['train']
test_dir = cfg.config['dir']['test']
image_dir = cfg.config['dir']['image']
initial_weight_path = cfg.config['path']['initial_weight']


# Net
net = model.YOLOv3(num_channels, num_classes)
if cuda:
    net = net.cuda()
net = net.train(True)
if initial_weight_path:
    net.load_state_dict(torch.load(initial_weight_path))


# Loss Function
criterion = nn.MSELoss()
if cuda:
    criterion = criterion.cuda()


# Optimizer
optimizer = optim.Adam(net.parameters())


# Dataset
def load_images(image_dir, data_dir):
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


def load_bboxes(data_dir):
    bboxes = os.listdir(data_dir)
    bboxes = [data_dir + data_file for data_file in bboxes]
    bboxes = [pre.load_bbox(path, height, width, cuda) for path in bboxes]

    return bboxes


train_images = load_images(image_dir, train_dir)
train_bboxes = load_bboxes(train_dir)
test_images = load_images(image_dir, test_dir)
test_bboxes = load_bboxes(test_dir)


# Train
def train_core(images, bboxes):
    num_images = images.size(0)
    detected_bboxes = net(images)
    detected_bboxes = post.postprocess(detected_bboxes,
                                       anchors,
                                       height,
                                       width,
                                       objectness,
                                       nms_iou,
                                       cuda)
