import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def tokenize(text):
    return re.sub(r'[^a-z ]', '', text.lower()).split()


def build_vocab(tokenized):
    all_words = [w for toks in tokenized for w in toks]
    vocab     = ['<PAD>', '<UNK>'] + [w for w, _ in Counter(all_words).most_common()]
    word2idx  = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx, all_words


def print_stats(texts, labels, tokenized, all_words):
    lengths = [len(t) for t in tokenized]
    print(f'Total samples  : {len(texts)}')
    print(f'Positive       : {sum(labels)}')
    print(f'Negative       : {len(labels) - sum(labels)}')
    print(f'Vocab size     : {len(set(all_words))}')
    print(f'Avg length     : {np.mean(lengths):.1f} words')
    print(f'Min / Max      : {min(lengths)} / {max(lengths)} words')


def plot_eda(labels, tokenized, save_path='./3assign/utils/eda.png'):
    lengths     = [len(t) for t in tokenized]
    pos_lengths = [l for l, lb in zip(lengths, labels) if lb == 1]
    neg_lengths = [l for l, lb in zip(lengths, labels) if lb == 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('EDA — IMDb Sentiment Dataset', fontsize=14, fontweight='bold')

    counts = [sum(labels), len(labels) - sum(labels)]
    bars   = axes[0].bar(['Positive', 'Negative'], counts,
                          color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.45)
    axes[0].set_title('Class Distribution')
    axes[0].set_ylabel('Number of Reviews')
    axes[0].set_ylim(0, max(counts) * 1.15)
    for bar, val in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, str(val), ha='center', fontweight='bold')

    axes[1].hist(pos_lengths, bins=20, alpha=0.7, color='#2ecc71',
                 label='Positive', edgecolor='white')
    axes[1].hist(neg_lengths, bins=20, alpha=0.7, color='#e74c3c',
                 label='Negative', edgecolor='white')
    axes[1].axvline(np.mean(lengths), color='navy', linestyle='--', linewidth=1.5,
                    label=f'Mean = {np.mean(lengths):.1f}')
    axes[1].set_title('Review Length Distribution')
    axes[1].set_xlabel('Number of Words')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()