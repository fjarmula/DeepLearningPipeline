import torch
import yaml
import os

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