import torch
import yaml
import os
import argparse
from model import SimpleCNN


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if
    torch.backends.mps.is_available() else "cpu")

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state, directory, filename="checkpoint.pth.tar"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, filename)
    torch.save(state, path)


def get_args():
    parser = argparse.ArgumentParser(description='MNIST CNN Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--grid_search', action='store_true')

    # Overrides for grid search parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument("--weight_decay", type=float)
    return parser.parse_args()


def get_optimizer(model, opt_name, lr, wd):
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Optimizer {opt_name} not supported")