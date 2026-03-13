import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import itertools
from model import SimpleCNN
from utils import get_args, get_device, load_config, get_optimizer, save_checkpoint
from preprocessing import get_dataloaders
from train import train_model


def main():
    args = get_args()
    device = get_device()
    config = load_config(args.config)

    if args.grid_search:
        lrs = config['training']['param_grid']['learning_rate']
        bss = config['training']['param_grid']['batch_size']
        opts = [args.optimizer] if args.optimizer else ["adam", "sgd"]
        wds = [args.weight_decay] if args.weight_decay is not None else [0.0, 1e-4] # make sure to include 0.0 for no weight decay
    else: # standard training, using list for uniformity in the loop
        lrs = [args.lr or config['training']['learning_rate']]
        bss = [args.batch_size or config['training']['batch_size']]
        opts = [args.optimizer or config['training']['optimizer']]
        wds = [args.weight_decay if args.weight_decay is not None else config['training']['weight_decay']]

    global_best_acc = 0.0
    best_params = None
    epochs = args.epochs or config['training']['epochs']
    criterion = nn.CrossEntropyLoss()

    total_experiments = len(lrs) * len(bss) * len(opts) * len(wds)
    print(f"\n{'=' * 60}")
    print(f"Starting {'Grid Search' if args.grid_search else 'Standard Training'}")
    print(f"Total Experiments: {total_experiments}")
    print(f"Device: {device} | Epochs: {epochs}")
    print(f"{'=' * 60}\n")

    # itertools.product creates every possible combination of parameters
    for lr, bs, opt_name, wd in itertools.product(lrs, bss, opts, wds):

        config['training']['batch_size'] = bs
        train_loader, test_loader = get_dataloaders(config)

        model = SimpleCNN().to(device)
        optimizer = get_optimizer(model, opt_name, lr, wd)

        exp_name = f"lr{lr}_bs{bs}_{opt_name}_wd{wd}"
        writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{exp_name}")

        print(f"Testing: LR={lr}, BS={bs}, Optimizer={opt_name}, WD={wd}")
        print("-" * 40)

        run_acc = train_model(
            model=model,
            epochs=epochs,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            config=config,
            filename=f"best_{exp_name}.pth.tar"
        )

        if run_acc > global_best_acc:
            global_best_acc = run_acc
            best_params = {
                'learning_rate': lr,
                'batch_size': bs,
                'optimizer': opt_name,
                'weight_decay': wd
            }

            print(f"New Global Best! Accuracy: {global_best_acc:.2f}%")
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'best_acc': global_best_acc,
                'params': best_params
            }, config['training']['checkpoint_dir'], filename="absolute_best_model.pth.tar")

        writer.close()
        print(f"Done with: {exp_name}\n")

    # 6. Final Summary
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("-" * 60)
    print(f"Best Accuracy: {global_best_acc:.2f}%")
    print(f"Best Parameters: {best_params}")
    print(f"Top Model Saved to: {config['training']['checkpoint_dir']}/absolute_best_model.pth.tar")
    print(f"Visualize all runs with: tensorboard --logdir={config['training']['log_dir']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()