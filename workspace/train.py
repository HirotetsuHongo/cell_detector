import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model import YOLOv3


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
    y1, y2, y3 = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
