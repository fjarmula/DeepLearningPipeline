import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import itertools
from model import *
from utils import *
from preprocessing import load_datasets, get_dataloaders
from train import train_model


def main():
    args = get_args()
    device = get_device()
    config = load_config(args.config)

    architectures = [
        {"name": "SimpleCNN", "type": "standard"},
        {"name": "Baseline", "type": "exp", "act": "relu", "bn": False, "drop": 0.0, "ks": 3},
        {"name": "Stabilized", "type": "exp", "act": "relu", "bn": True, "drop": 0.3, "ks": 3},
        {"name": "High-Vision", "type": "exp", "act": "relu", "bn": False, "drop": 0.0, "ks": 5},
        {"name": "Modernist", "type": "exp", "act": "gelu", "bn": False, "drop": 0.0, "ks": 3},
    ]

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

    train_subset, test_subset = load_datasets(config)

    active_archs = [a for a in architectures if not args.model or a['name'].lower() == args.model.lower()]

    hyper_combinations = len(list(itertools.product(lrs, bss, opts, wds)))
    total_runs = len(active_archs) * hyper_combinations
    current_run_idx = 0

    global_best_acc = 0.0
    best_overall_config = None
    epochs = args.epochs or config['training']['epochs']
    criterion = nn.CrossEntropyLoss()
    timed_train_model = measure_time(train_model)

    print("-" * 60)
    print(f"Starting Experiment Session")
    print(f"Device: {device} | Total Runs: {total_runs}")
    print(f"Subset Sizes: Train={len(train_subset)}, Test={len(test_subset)}")
    print("-" * 60)

    for arch in active_archs:
        arch_best_acc = 0.0
        print(f"\nArchitecture: {arch['name']}")

        for lr, bs, opt_name, wd in itertools.product(lrs, bss, opts, wds):
            current_run_idx += 1

            train_loader, test_loader = get_dataloaders(train_subset, test_subset, bs)

            if arch['type'] == "standard":
                model = SimpleCNN().to(device)
            else:
                model = ExperimentalCNN(
                    activation=arch['act'],
                    use_batchnorm=arch['bn'],
                    dropout_p=arch['drop'],
                    kernel_size=arch['ks']
                ).to(device)

            optimizer = get_optimizer(model, opt_name, lr, wd)
            param_count = sum(p.numel() for p in model.parameters())

            exp_name = f"{arch['name']}_lr{lr}_bs{bs}_{opt_name}_wd{wd}"

            print(f"\nRun [{current_run_idx}/{total_runs}]: Testing: lr={lr}, bs={bs}, optimizer={opt_name}, wd={wd}")
            print(f"Parameters: {param_count:,}")

            writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")
            writer.add_scalar("Config/ParamCount", param_count)

            (run_acc, run_best_weights, convergence_epoch), duration = timed_train_model(
                model=model,
                epochs=epochs,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                writer=writer,
            )

            if run_acc > arch_best_acc:
                arch_best_acc = run_acc
                arch_filename = f'best_{arch["name"]}.pth.tar'
                save_checkpoint({
                    'model_state_dict': run_best_weights,
                    'acc': arch_best_acc,
                    'arch': arch,
                    'params': {'lr': lr, 'bs': bs, 'opt': opt_name, 'wd': wd}
                }, config['training']['checkpoint_dir'], filename=arch_filename)

            if run_acc > global_best_acc:
                global_best_acc = run_acc
                best_overall_config = {
                    "arch": arch['name'],
                    "lr": lr,
                    "bs": bs,
                    "opt": opt_name,
                    "wd": wd
                }
                save_checkpoint({
                    'model_state_dict': run_best_weights,
                    'acc': global_best_acc,
                    'config': best_overall_config
                }, config['training']['checkpoint_dir'], filename='best_overall_model.pth.tar')

            writer.close()
            print(f"\nRun {current_run_idx} Completed | Accuracy: {run_acc:.2f}% | Time: {duration:.2f}s | Convergence epoch: {convergence_epoch}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("-" * 60)
    print(f"Best Overall Accuracy: {global_best_acc:.2f}%")
    print(f"Winner Config: {best_overall_config}")
    print(f"Visualize all runs with: tensorboard --logdir={log_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()