import torch
import itertools
import copy
from utils import measure_time

from torch.ao.nn.quantized.functional import threshold

from utils import save_checkpoint

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()

    return running_loss / len(loader), 100 * correct / len(loader.dataset)

def validate(model, device, loader, criterion):
    model.eval()
    val_loss = 0.0
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return val_loss/len(loader), 100 * correct / total

@measure_time
def train_model(model, epochs, device, train_loader, test_loader, optimizer, criterion, writer):
    run_best_acc = 0.0
    best_model_state = None

    prev_loss = float('inf')
    convergence_epoch = epochs
    threshold = 0.001
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, test_loader, criterion)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        loss_diff = abs(prev_loss - train_loss)
        if loss_diff < threshold:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= 3 and convergence_epoch == epochs: # I do not implement early stopping as for now se we need to train for all epochs
            # to compare the results, but track the epoch of convergence.
            convergence_epoch = epoch + 1

        prev_loss = train_loss

        if val_acc > run_best_acc:
            run_best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())


        print(f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return run_best_acc, best_model_state, convergence_epoch



