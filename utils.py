import torch
import yaml
import os
import argparse

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

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Override epochs in ''config')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--grid_search', action='store_true', help='Run grid search')

    return parser.parse_args()