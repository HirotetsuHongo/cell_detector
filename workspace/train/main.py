import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model import YOLOv3


# Model
model = YOLOv3(config['channels']).cuda()
model = model.train(True)
model.load_state_dict(torch.load(config['path']['initial_parameter']))

criterion = nn.MSELoss
optimizer = optim.SGD(model.parameters(),
                      lr=config['learning_rate'],
                      momentum=config['momentum'])

# Dataset

# Train
