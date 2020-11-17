import torch
import yaml


config = {}

with open('/workspace/config/config.yaml') as yaml_file:
    config = yaml_file.read()
    config = yaml.load(config, Loader=yaml.SafeLoader)

if config['CUDA'] == 'Enable':
    assert torch.cuda.is_available() is True
    config['device'] = torch.device('cuda')
else:
    config['device'] = torch.device('cpu')
