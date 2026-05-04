"""
train_part2.py — Part 2: Scratch CNN vs VGG Transfer Learning on CIFAR-10
Run: python train_part2.py --model scratch
     python train_part2.py --model vgg
     python train_part2.py --model both   (trains both sequentially)
"""

import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from cifar10_models import ScratchCNN, get_vgg_transfer, unfreeze_vgg_top_layers
from plots import plot_training_curves, plot_comparison


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 64
NUM_CLASSES = 10
NUM_WORKERS = 2

SCRATCH_EPOCHS   = 15
VGG_HEAD_EPOCHS  = 3   # train only the head first
VGG_FINETUNE_EPOCHS = 4  # then fine-tune top conv blocks


# ─────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────
def get_dataloaders(model_type='scratch'):
    """
    WHY different transforms per model:
    - Scratch CNN: 32x32 input (native CIFAR-10 size)
    - VGG: expects 224x224 (ImageNet size), so we resize up.
      This is the standard approach for applying pre-trained ImageNet
      models to small-image datasets.
    """
    if model_type == 'vgg':
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),   # ImageNet mean
                                 (0.229, 0.224, 0.225)),  # ImageNet std
        ])
        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 mean
                                 (0.2023, 0.1994, 0.2010)), # CIFAR-10 std
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True,
                                               transform=train_transform)
    test_set   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True,
                                               transform=val_transform)

    # 80/20 train/val split
    train_size = int(0.8 * len(full_train))
    val_size   = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return total_loss / total, 100. * correct / total


def run_training(model, train_loader, val_loader, optimizer, scheduler,
                 criterion, epochs, label):
    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    start_time    = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"[{label}] Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"\n[{label}] Training done in {elapsed/60:.1f} min | "
          f"Best Val Acc: {best_val_acc:.2f}%\n")

    model.load_state_dict(best_weights)  # restore best
    return model, history, elapsed


# ─────────────────────────────────────────────
#  Train Scratch CNN
# ─────────────────────────────────────────────
def train_scratch():
    print("=" * 60)
    print("PART 2a — Training CNN from Scratch")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders('scratch')

    model     = ScratchCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=SCRATCH_EPOCHS)

    model, history, elapsed = run_training(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        SCRATCH_EPOCHS, label='Scratch'
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[Scratch] Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), 'scratch_cnn.pth')
    plot_training_curves(history, title='Scratch CNN', save_path='scratch_curves.png')

    return history, test_acc, elapsed


# ─────────────────────────────────────────────
#  Train VGG Transfer Learning
# ─────────────────────────────────────────────
def train_vgg():
    print("=" * 60)
    print("PART 2b — VGG16 Transfer Learning")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders('vgg')

    # Stage 1: train head only (backbone frozen)
    print("\n--- Stage 1: Training classifier head (backbone frozen) ---")
    model     = get_vgg_transfer(num_classes=NUM_CLASSES, freeze_features=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # WHY different lr: head has random weights → needs higher lr
    # frozen backbone → optimizer only sees classifier params
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model, history_head, elapsed_head = run_training(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        VGG_HEAD_EPOCHS, label='VGG-Head'
    )

    # Stage 2: unfreeze top conv blocks and fine-tune with lower lr
    print("\n--- Stage 2: Fine-tuning top conv blocks ---")
    model = unfreeze_vgg_top_layers(model, n_blocks=1)

    # WHY lower lr for fine-tuning: pre-trained weights are valuable,
    # high lr would destroy them. Small updates let features adapt gently.
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=VGG_FINETUNE_EPOCHS)

    model, history_ft, elapsed_ft = run_training(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        VGG_FINETUNE_EPOCHS, label='VGG-Finetune'
    )

    # Merge histories for plotting
    history = {
        'train_loss': history_head['train_loss'] + history_ft['train_loss'],
        'val_loss':   history_head['val_loss']   + history_ft['val_loss'],
        'train_acc':  history_head['train_acc']  + history_ft['train_acc'],
        'val_acc':    history_head['val_acc']    + history_ft['val_acc'],
    }

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"[VGG] Test Accuracy: {test_acc:.2f}%")

    torch.save(model.state_dict(), 'vgg_transfer.pth')
    plot_training_curves(history, title='VGG16 Transfer Learning',
                         save_path='vgg_curves.png',
                         vline_at=VGG_HEAD_EPOCHS)  # mark stage boundary

    elapsed = elapsed_head + elapsed_ft
    return history, test_acc, elapsed


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['scratch', 'vgg', 'both'],
                        default='both')
    args = parser.parse_args()

    print(f"\nUsing device: {DEVICE}\n")

    scratch_history = vgg_history = None
    scratch_acc     = vgg_acc     = None

    if args.model in ('scratch', 'both'):
        scratch_history, scratch_acc, scratch_time = train_scratch()

    if args.model in ('vgg', 'both'):
        vgg_history, vgg_acc, vgg_time = train_vgg()

    if args.model == 'both':
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  Scratch CNN  — Test Acc: {scratch_acc:.2f}% | "
              f"Time: {scratch_time/60:.1f} min")
        print(f"  VGG Transfer — Test Acc: {vgg_acc:.2f}%  | "
              f"Time: {vgg_time/60:.1f} min")
        plot_comparison(scratch_history, vgg_history,
                        scratch_acc, vgg_acc,
                        save_path='comparison.png')