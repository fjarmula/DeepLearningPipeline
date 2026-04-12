import torch
from torch import nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ExperimentalCNN(nn.Module):
    def __init__(self, activation='relu', use_batchnorm=False, dropout_p=0.0, kernel_size=3):
        super().__init__()
        acts = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        self.act = acts.get(activation.lower())
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()

        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
