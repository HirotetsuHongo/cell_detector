import torch
import yaml


config = {}

with open('config/config.yaml') as yaml_file:
    config = yaml_file.read()
    config = yaml.load(config, Loader=yaml.SafeLoader)

if config['CUDA']:
    assert torch.cuda.is_available() is True
