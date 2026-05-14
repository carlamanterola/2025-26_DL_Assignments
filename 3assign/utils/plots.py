"""
plots.py
========
Plotting utilities for LSTM sentiment classification results.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_losses: list, test_losses: list,
                         train_accs: list, test_accs: list):
    """
    Two-panel plot: training/test loss and training/test accuracy per epoch.
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Sentiment Classifier — Training Results", fontsize=14, fontweight='bold')

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_losses, color='steelblue',  linewidth=1.8, marker='o',
            markersize=4, label='Train loss')
    ax.plot(epochs, test_losses,  color='darkorange', linewidth=1.8, marker='s',
            markersize=4, label='Test loss')
    ax.set_title("Loss per Epoch (BCE)", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_xticks(epochs)
    ax.grid(axis='y', alpha=0.3)

    # Best test loss annotation
    best_epoch = int(np.argmin(test_losses)) + 1
    best_loss  = min(test_losses)
    ax.annotate(f'Best: {best_loss:.4f}',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 0.3, best_loss + 0.005),
                arrowprops=dict(arrowstyle='->', color='darkorange'),
                fontsize=8, color='darkorange')

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in train_accs], color='steelblue',  linewidth=1.8,
            marker='o', markersize=4, label='Train acc')
    ax.plot(epochs, [a * 100 for a in test_accs],  color='darkorange', linewidth=1.8,
            marker='s', markersize=4, label='Test acc')
    ax.set_title("Accuracy per Epoch (%)", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    best_epoch_acc = int(np.argmax(test_accs)) + 1
    best_acc       = max(test_accs) * 100
    ax.annotate(f'Best: {best_acc:.2f}%',
                xy=(best_epoch_acc, best_acc),
                xytext=(best_epoch_acc + 0.3, best_acc - 5),
                arrowprops=dict(arrowstyle='->', color='darkorange'),
                fontsize=8, color='darkorange')

    plt.tight_layout()
    plt.show()


def plot_final_summary(train_losses: list, test_losses: list,
                       train_accs: list, test_accs: list):
    """
    Single summary bar chart of final epoch train vs test metrics.
    """
    labels  = ['Loss', 'Accuracy (%)']
    train_v = [train_losses[-1], train_accs[-1] * 100]
    test_v  = [test_losses[-1],  test_accs[-1]  * 100]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, train_v, w, color='steelblue',  alpha=0.85, label='Train')
    ax.bar(x + w/2, test_v,  w, color='darkorange', alpha=0.85, label='Test')

    for i, (tv, ev) in enumerate(zip(train_v, test_v)):
        ax.text(i - w/2, tv + 0.5, f'{tv:.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, ev + 0.5, f'{ev:.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Final Epoch — Train vs Test Metrics", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()