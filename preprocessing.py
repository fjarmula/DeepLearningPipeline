import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.v2 import RandomAffine


def get_transform(aug_type='standard'):
    if aug_type == 'augmented':
        return transforms.Compose([transforms.RandomRotation(30),
                                   RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                   ])
    else:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])


def load_datasets(config, transform_type='standard'):
    transform = get_transform(transform_type)

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(config['data']['subset_train']))
    test_subset = Subset(test_dataset, range(config['data']['subset_test']))

    return train_subset, test_subset


def get_dataloaders(train_subset, test_subset, batch_size):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader