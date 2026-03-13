import argparse
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import SimpleCNN
from utils import get_args
from utils import save_checkpoint
from utils import get_device, load_config
from preprocessing import get_dataloaders
from train import train_one_epoch, validate, train_model

def main():
    args = get_args()
    device = get_device()

    config = load_config(args.config)
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    global_best_acc = 0.0
    best_params = None

    epochs = config['training']['epochs']
    criterion = nn.CrossEntropyLoss()

    if not args.grid_search:
        train_loader, test_loader = get_dataloaders(config)
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        writer = SummaryWriter(log_dir=config['training']['log_dir'])
        print(f"Starting standard training for {epochs} epochs on device {device}...")
        train_model(model, epochs, device, train_loader, test_loader, optimizer, criterion, writer, config)
        writer.close()

    if args.grid_search:

        lrs = config['training']['param_grid']['learning_rate']
        bss = config['training']['param_grid']['batch_size']

        print(f"\n Starting grid search on device {device}...")

        for lr, bs in itertools.product(lrs, bss):
            config['training']['learning_rate'] = lr
            config['training']['batch_size'] = bs

            train_loader, test_loader = get_dataloaders(config)
            model = SimpleCNN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            exp_name = f"lr_{lr}_bs_{bs}"
            writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{exp_name}")

            print(f"\nTesting: LR={lr}, Batch Size={bs}")
            print("-" * 50)

            run_acc = train_model(model, epochs, device, train_loader, test_loader, optimizer, criterion, writer, config, filename=f"best_{exp_name}.pth.tar")

            if run_acc > global_best_acc:
                global_best_acc = run_acc
                best_params = {'learning_rate': lr, 'batch_size': bs}

                print(f"New Global Best: {global_best_acc:.2f}% with LR={lr}, BS={bs}")
                save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'best_acc': global_best_acc,
                    'params': {'lr': lr, 'bs': bs}
                }, config['training']['checkpoint_dir'], filename="absolute_best_model.pth.tar")
            writer.close()

        print("\n" + "="*50)
        print(f"Best Accuracy: {global_best_acc:.2f}%")
        print(f"Best Parameters: {best_params}")
        print(f"Saved to: {config['training']['checkpoint_dir']}/absolute_best_model.pth.tar")
        print("="*50)

if __name__ == "__main__":
    main()