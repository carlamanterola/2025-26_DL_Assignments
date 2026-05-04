"""
utils/plots.py — Training curve and comparison plots
"""

import matplotlib.pyplot as plt


def plot_training_curves(history, title='Model', save_path=None, vline_at=None):
    """Plot loss and accuracy curves for a single model."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'],   label='Val')
    if vline_at:
        ax1.axvline(x=vline_at, color='gray', linestyle='--',
                    label='Fine-tune starts')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['val_acc'],   label='Val')
    if vline_at:
        ax2.axvline(x=vline_at, color='gray', linestyle='--',
                    label='Fine-tune starts')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    plt.show()


def plot_comparison(scratch_history, vgg_history,
                    scratch_acc, vgg_acc, save_path=None):
    """Side-by-side comparison of both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Scratch CNN vs VGG Transfer Learning — CIFAR-10',
                 fontsize=14, fontweight='bold')

    # Val accuracy comparison
    ax = axes[0]
    ax.plot(range(1, len(scratch_history['val_acc']) + 1),
            scratch_history['val_acc'], label=f'Scratch CNN (test: {scratch_acc:.1f}%)',
            color='steelblue')
    ax.plot(range(1, len(vgg_history['val_acc']) + 1),
            vgg_history['val_acc'],    label=f'VGG Transfer (test: {vgg_acc:.1f}%)',
            color='tomato')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val loss comparison
    ax = axes[1]
    ax.plot(range(1, len(scratch_history['val_loss']) + 1),
            scratch_history['val_loss'], label='Scratch CNN', color='steelblue')
    ax.plot(range(1, len(vgg_history['val_loss']) + 1),
            vgg_history['val_loss'],     label='VGG Transfer', color='tomato')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot: {save_path}")
    plt.show()