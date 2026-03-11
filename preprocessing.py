import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True,
                                  transform=transform)

    train_subset = Subset(train_dataset, range(config['data'][
                                                   'subset_train']))
    test_subset = Subset(test_dataset, range(config['data']['subset_test']))

    train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'],
                              shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=config['training']['batch_size'],
                             shuffle=False)

    return train_loader, test_loader