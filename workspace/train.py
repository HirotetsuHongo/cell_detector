import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model import YOLOv3


# Model
model = YOLOv3(config['channels']).cuda()
model = model.train(True)
if config['path']['initial_weight'] != False:
    model.load_state_dict(torch.load(config['path']['initial_weight']))

criterion = nn.MSELoss
optimizer = optim.Adam(model.parameters())

# Dataset

# Train
