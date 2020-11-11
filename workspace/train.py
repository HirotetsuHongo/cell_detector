import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model import YOLOv3
from postprocess import transform_feature_map


# Model
model = YOLOv3(config['channels'], config['classes']).cuda()
model = model.train(True)
if config['path']['initial_weight'] != False:
    model.load_state_dict(torch.load(config['path']['initial_weight']))

criterion = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters())

# Dataset

# Train
def train_step(x, target, model, criterion, optimizer):
    optimizer.zero_grad()
    height, width = x.size()
    assert height == width
    size = height
    y1, y2, y3 = model(x)
    y1, y2, y3 = [transform_feature_map(y, size, anchors[0], anchors[1], anchors[2])
                  for y, anchors in [(y1, config['anchors'][0]),
                                     (y2, config['anchors'][1]),
                                     (y3, config['anchors'][2])]]
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
