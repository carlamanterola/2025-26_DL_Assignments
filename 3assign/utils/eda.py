"""
eda.py
======
Exploratory Data Analysis for the IMDB sentiment dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def plot_imdb_eda(train_raw, test_raw):
    # ── Label distribution ────────────────────────────────────────────────────
    train_labels = [item["label"] for item in train_raw]
    test_labels  = [item["label"] for item in test_raw]

    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    test_pos  = sum(test_labels)
    test_neg  = len(test_labels) - test_pos

    # ── Review length distribution ────────────────────────────────────────────
    train_lengths = [len(tokenize(item["text"])) for item in train_raw]
    test_lengths  = [len(tokenize(item["text"])) for item in test_raw]

    train_pos_lens = [l for l, lbl in zip(train_lengths, train_labels) if lbl == 1]
    train_neg_lens = [l for l, lbl in zip(train_lengths, train_labels) if lbl == 0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IMDB Dataset — Exploratory Data Analysis", fontsize=15, fontweight='bold')

    colors = {'pos': '#2ecc71', 'neg': '#e74c3c', 'train': 'steelblue', 'test': 'darkorange'}

    # ── Plot 1: Label counts (train vs test) ──────────────────────────────────
    ax = axes[0]
    x  = np.arange(2)
    w  = 0.35
    bars_train = ax.bar(x - w/2, [train_neg, train_pos], w,
                        color=[colors['neg'], colors['pos']], alpha=0.85, label='Train')
    bars_test  = ax.bar(x + w/2, [test_neg,  test_pos],  w,
                        color=[colors['neg'], colors['pos']], alpha=0.5,  label='Test',
                        edgecolor='black', linewidth=0.8)
    for bar in list(bars_train) + list(bars_test):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_title("Label Distribution", fontsize=12)
    ax.set_ylabel("Count")
    neg_patch   = mpatches.Patch(color=colors['neg'],  label='Negative')
    pos_patch   = mpatches.Patch(color=colors['pos'],  label='Positive')
    train_patch = mpatches.Patch(color='gray', alpha=0.85, label='Train')
    test_patch  = mpatches.Patch(color='gray', alpha=0.5,  label='Test')
    ax.legend(handles=[neg_patch, pos_patch, train_patch, test_patch], fontsize=8)
    ax.set_ylim(0, max(train_neg, train_pos, test_neg, test_pos) * 1.15)

    # ── Plot 2: Review length histogram (train) ───────────────────────────────
    ax = axes[1]
    bins = np.linspace(0, 1500, 60)
    ax.hist(train_neg_lens, bins=bins, color=colors['neg'], alpha=0.6, label='Negative')
    ax.hist(train_pos_lens, bins=bins, color=colors['pos'], alpha=0.6, label='Positive')
    ax.axvline(np.median(train_neg_lens), color=colors['neg'], linestyle='--',
               linewidth=1.2, label=f'Neg median ({int(np.median(train_neg_lens))})')
    ax.axvline(np.median(train_pos_lens), color=colors['pos'], linestyle='--',
               linewidth=1.2, label=f'Pos median ({int(np.median(train_pos_lens))})')
    ax.set_title("Review Length Distribution (Train)", fontsize=12)
    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # ── Plot 3: CDF of review lengths ─────────────────────────────────────────
    ax = axes[2]
    for lengths, label, color in [
        (train_lengths, 'Train', colors['train']),
        (test_lengths,  'Test',  colors['test']),
    ]:
        sorted_l = np.sort(lengths)
        cdf      = np.arange(1, len(sorted_l) + 1) / len(sorted_l)
        ax.plot(sorted_l, cdf, color=color, linewidth=1.5, label=label)

    ax.axvline(200, color='gray', linestyle='--', linewidth=1.2, label='Truncation (200)')
    pct_train_under = np.mean(np.array(train_lengths) <= 200) * 100
    ax.text(210, 0.05, f'{pct_train_under:.1f}% ≤ 200\ntokens (train)',
            fontsize=8, color='gray')
    ax.set_title("CDF of Review Lengths", fontsize=12)
    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Cumulative proportion")
    ax.set_xlim(0, 1500)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Loading IMDB dataset …")
    raw      = load_dataset("imdb")
    train_raw = raw["train"]
    test_raw  = raw["test"]
    plot_imdb_eda(train_raw, test_raw)