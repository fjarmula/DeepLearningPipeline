import torch
import yaml
import os
import argparse
import numpy as np
import random
import time
from functools import wraps
from model import SimpleCNN


def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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
    parser.add_argument('--logdir', type=str)

    parser.add_argument('--lr', nargs='+', type=float, help='List of learning rates')
    parser.add_argument('--batch_size', nargs='+', type=int, help='List of batch sizes')
    parser.add_argument('--optimizer', nargs='+', type=str, help='List of optimizers (e.g. adam sgd)')
    parser.add_argument("--weight_decay", nargs='+', type=float, help='List of weight decay values')
    parser.add_argument('--model', type=str, help='Specific model to run (e.g., SimpleCNN, Stabilized)')
    parser.add_argument('--seed', nargs='+', type=int, help='List of random seeds')

    return parser.parse_args()


def get_optimizer(model, opt_name, lr, wd):
    opt_name = opt_name.lower()
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Optimizer {opt_name} not supported")


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start

    return wrapper

def get_architectures():
    return [
        {"name": "SimpleCNN", "type": "standard"},
        {"name": "Baseline", "type": "exp", "act": "relu", "bn": False, "drop": 0.0, "ks": 3},
        {"name": "Stabilized", "type": "exp", "act": "relu", "bn": True, "drop": 0.3, "ks": 3},
        {"name": "High-Vision", "type": "exp", "act": "relu", "bn": False, "drop": 0.0, "ks": 5},
        {"name": "Modernist", "type": "exp", "act": "gelu", "bn": False, "drop": 0.0, "ks": 3},
    ]

def prepare_training_params(config, args):
    seeds = args.seed or [None, None, None, 42, 42, 42]
    if args.grid_search:
        lrs = args.lr or config['training']['param_grid']['learning_rate']
        bss = args.batch_size or config['training']['param_grid']['batch_size']
        opts = args.optimizer or ["adam", "sgd"]
        wds = args.weight_decay if args.weight_decay is not None else [0.0, 1e-4]
        log_dir = args.logdir or config['training']['log_dir_grid']
    else:
        lrs = [args.lr[0]] if args.lr else [config['training']['learning_rate']]
        bss = [args.batch_size[0]] if args.batch_size else [config['training']['batch_size']]
        opts = [args.optimizer[0]] if args.optimizer else [config['training']['optimizer']]
        wds = [args.weight_decay[0]] if args.weight_decay is not None else [config['training']['weight_decay']]
        log_dir = args.logdir or config['training']['log_dir']

    return lrs, bss, opts, wds, log_dir, seeds
