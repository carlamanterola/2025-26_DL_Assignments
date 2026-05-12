import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data.dataset import ReviewDataset, collate_fn, encode
from models.lstm import SentimentLSTM
from utils.eda import tokenize, build_vocab, print_stats, plot_eda
from utils.plots import plot_training_curves, plot_confusion_matrix


def run(texts, labels, config):
    device     = config['device']
    seed       = config['seed']
    epochs     = config['epochs']
    batch_size = config['batch_size']
    embed_dim  = config['embed_dim']
    hidden_dim = config['hidden_dim']
    dropout    = config['dropout']
    lr         = config['lr']

    # ── Preprocessing ─────────────────────────────────────────────────────────
    tokenized            = [tokenize(t) for t in texts]
    vocab, word2idx, all_words = build_vocab(tokenized)
    vocab_size           = len(vocab)

    print_stats(texts, labels, tokenized, all_words)
    plot_eda(labels, tokenized)

    encoded = [encode(t, word2idx) for t in tokenized]

    # ── Split ─────────────────────────────────────────────────────────────────
    indices = list(range(len(texts)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.25, stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed
    )
    print(f'Train: {len(train_idx)}  |  Val: {len(val_idx)}  |  Test: {len(test_idx)}')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dl = DataLoader(ReviewDataset(train_idx, encoded, labels),
                          batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_dl   = DataLoader(ReviewDataset(val_idx,   encoded, labels),
                          batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(ReviewDataset(test_idx,  encoded, labels),
                          batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = SentimentLSTM(vocab_size, embed_dim, hidden_dim, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Trainable parameters: {n_params:,}')

    # ── Training ──────────────────────────────────────────────────────────────
    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, n_correct, n_total = 0.0, 0, 0

        with (torch.enable_grad() if training else torch.no_grad()):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = criterion(logits, yb)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                preds      = (torch.sigmoid(logits) >= 0.5).float()
                n_correct  += (preds == yb).sum().item()
                n_total    += yb.size(0)
                total_loss += loss.item() * yb.size(0)

        return total_loss / n_total, n_correct / n_total

    history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_dl, training=True)
        vl_loss, vl_acc = run_epoch(val_dl,   training=False)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), './3assign/models/best_model_sentiment.pt')

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{epochs}  '
                  f'| Train  loss: {tr_loss:.4f}  acc: {tr_acc:.0%}  '
                  f'| Val  loss: {vl_loss:.4f}  acc: {vl_acc:.0%}')

    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')
    plot_training_curves(history, epochs)

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load('best_model_sentiment.pt', map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = (torch.sigmoid(model(xb.to(device))) >= 0.5).long().cpu()
            all_preds.extend(preds.tolist())
            all_targets.extend(yb.long().tolist())

    print('=' * 52)
    print('TEST SET RESULTS')
    print('=' * 52)
    print(classification_report(all_targets, all_preds,
                                 target_names=['Negative (0)', 'Positive (1)']))
    plot_confusion_matrix(all_targets, all_preds)

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(review_text):
        model.eval()
        tokens = tokenize(review_text)
        tensor = encode(tokens, word2idx).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()

        label      = 'POSITIVE' if prob >= 0.5 else 'NEGATIVE'
        confidence = max(prob, 1 - prob)
        print(f'  [{label}] conf={confidence:.1%}  p={prob:.3f}  |  "{review_text}"')

    return predict