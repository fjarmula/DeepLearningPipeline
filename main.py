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

    architectures = get_architectures()
    lrs, bss, opts, wds, log_dir, seeds = prepare_training_params(config, args)
    train_subset, test_subset = load_datasets(config, transform_type=args.transform)

    active_archs = [a for a in architectures if not args.model or a['name'].lower() == args.model.lower()]

    experiments = list(itertools.product(seeds, active_archs, lrs, bss, opts, wds))

    total_runs = len(experiments)
    global_best_acc = 0.0
    best_overall_config = None
    epochs = args.epochs or config['training']['epochs']
    criterion = get_criterion(args.criterion) if args.criterion else get_criterion(config['training']['criterion'])

    print("-" * 60)
    print(f"Starting Experiment Session")
    print(f"Total Runs to Execute: {total_runs}")
    print(f'{criterion=}:')
    print("-" * 60)

    for idx, (seed, arch, lr, bs, opt_name, wd) in enumerate(experiments, 1):

        set_seed(seed)

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
        exp_name = f"{arch['name']}_lr{lr}_bs{bs}_{opt_name}_seed{seed}"

        print(f"\nRun [{idx}/{total_runs}] | Arch: {arch['name']} | Seed: {seed}")
        print(f"Params: {param_count:,} | LR: {lr} | BS: {bs} | Opt: {opt_name}")

        writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")

        (run_acc, run_best_weights, conv_epoch), duration = train_model(
            model=model, epochs=epochs, device=device,
            train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, criterion=criterion, writer=writer,
        )

        if run_acc > global_best_acc:
            global_best_acc = run_acc
            best_overall_config = {"arch": arch['name'], "lr": lr, "bs": bs, "seed": seed}
            save_checkpoint({
                'model_state_dict': run_best_weights,
                'acc': global_best_acc,
                'config': best_overall_config
            }, config['training']['checkpoint_dir'], filename='best_model_model.pth.tar')

        writer.close()
        print(f"Result: {run_acc:.2f}% | Time: {duration:.2f}s | Convergence epoch: {conv_epoch}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("-" * 60)
    print(f"Best Overall Accuracy: {global_best_acc:.2f}%")
    print(f"Winner Config: {best_overall_config}")
    print(f"Visualize all runs with: tensorboard --logdir={log_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()