import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import itertools
from model import SimpleCNN
from utils import *
from preprocessing import get_dataloaders
from train import train_model


def main():
    args = get_args()
    device = get_device()
    config = load_config(args.config)
    #set_seed(config['training']['seed'])

    if args.grid_search:
        lrs = args.lr if args.lr else config['training']['param_grid']['learning_rate']
        bss = args.batch_size if args.batch_size else config['training']['param_grid']['batch_size']
        opts = args.optimizer if args.optimizer else ["adam", "sgd"]
        wds = args.weight_decay if args.weight_decay is not None else [0.0, 1e-4]
        log_dir = args.logdir if args.logdir is not None else config['training']['log_dir_grid']
        filename = "best_model_grid.pth.tar"

    else:
        lrs = [args.lr[0]] if args.lr is not None else [config['training']['learning_rate']]
        bss = [args.batch_size[0]] if args.batch_size is not None else [config['training']['batch_size']]
        opts = [args.optimizer[0]] if args.optimizer is not None else [config['training']['optimizer']]
        wds = [args.weight_decay[0]] if args.weight_decay is not None else [config['training']['weight_decay']]
        log_dir = args.logdir if args.logdir is not None else config['training']['log_dir']
        filename = "best_model_standard.pth.tar"

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

    for i, (lr, bs, opt_name, wd) in enumerate(itertools.product(lrs, bss, opts, wds), 1):

        config['training']['batch_size'] = bs
        train_loader, test_loader = get_dataloaders(config)

        model = SimpleCNN().to(device)
        optimizer = get_optimizer(model, opt_name, lr, wd)

        exp_name = f"lr{lr}_bs{bs}_{opt_name}_wd{wd}"
        writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")

        print(f"[{i}/{total_experiments}] Testing: LR={lr}, BS={bs}, Optimizer={opt_name}, WD={wd}")
        print("-" * 40)

        run_acc, run_best_weights = train_model(
            model=model,
            epochs=epochs,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
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
                'model_state_dict': run_best_weights,
                'best_acc': global_best_acc,
                'params': best_params
            }, config['training']['checkpoint_dir'], filename=filename)

        writer.close()
        print()

    # 6. Final Summary
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("-" * 60)
    print(f"Best Accuracy: {global_best_acc:.2f}%")
    print(f"Best Parameters: {best_params}")
    print(f"Top Model Saved to: {config['training']['checkpoint_dir']}/{filename}")
    print(f"Visualize all runs with: tensorboard --logdir={log_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()