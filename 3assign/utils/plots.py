import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_training_curves(history, epochs, save_path='training_curves.png'):
    ep_range = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Training Curves — Sentiment LSTM', fontsize=14, fontweight='bold')

    axes[0].plot(ep_range, history['train_loss'], label='Train', color='#3498db', linewidth=2)
    axes[0].plot(ep_range, history['val_loss'],   label='Val',   color='#e74c3c', linewidth=2)
    axes[0].set_title('Loss (BCEWithLogitsLoss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_range, history['train_acc'], label='Train', color='#3498db', linewidth=2)
    axes[1].plot(ep_range, history['val_acc'],   label='Val',   color='#e74c3c', linewidth=2)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(all_targets, all_preds, save_path='confusion_matrix.png'):
    cm  = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Negative', 'Positive']
    ).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()