import numpy as np
import torch
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
from models import lstm as lstm_module
from utils.eda import plot_imdb_eda
from utils.plots import plot_training_curves, plot_final_summary
import models.baseline_pipeline as baseline
from models.TS_model import (
    load_and_clean,
    test_stationarity,
    plot_stationarity,
    build_sequences_from_segments,
    split_data,
    LSTMForecaster,
    train_model,
    predict,
    forecast_future,
    plot_training_loss,
    plot_predictions,
    plot_forecast,
)


# REPRODUCIBILITY
# ===========================================================================
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===========================================================================
# TASK 1: LANGUAGE - MOVIE REVIEW CLASSIFICATION
# ===========================================================================
print("=" * 80)
print("TASK 1: LANGUAGE - MOVIE REVIEW CLASSIFICATION")
print("=" * 80)

# ── Config ────────────────────────────────────────────────────────────────────
GLOVE_PATH = "./3assign/data/glove.6B.100d.txt"
EPOCHS_NLP = 8
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading IMDB dataset …")
raw       = load_dataset("imdb")
train_raw = raw["train"]
test_raw  = raw["test"]
 
print(f"Train size: {len(train_raw)} | Test size: {len(test_raw)}")
 
# ── EDA ───────────────────────────────────────────────────────────────────────
plot_imdb_eda(train_raw, test_raw)
 
# ── Vocabulary & encoding ─────────────────────────────────────────────────────
vocab, stoi = lstm_module.build_vocab(train_raw)
VOCAB_SIZE  = len(vocab)
PAD_IDX     = stoi["<pad>"]
UNK_IDX     = stoi["<unk>"]
encode_fn   = lstm_module.make_encoder(stoi, unk_idx=UNK_IDX)
 
# ── Dataloaders ───────────────────────────────────────────────────────────────
train_loader, test_loader = lstm_module.build_dataloaders(
    train_raw, test_raw, encode_fn, pad_idx=PAD_IDX,
    batch_size=lstm_module.BATCH_SIZE,
)
 
# ── Model ─────────────────────────────────────────────────────────────────────
model = lstm_module.LSTM(
    vocab_size=VOCAB_SIZE,
    embedding_dim=lstm_module.EMBEDDING_DIM,
    hidden_dim=lstm_module.HIDDEN_DIM,
    pad_idx=PAD_IDX,
)
 
try:
    pretrained = lstm_module.load_glove(GLOVE_PATH, stoi, pad_idx=PAD_IDX,
                                        dim=lstm_module.EMBEDDING_DIM)
    model.embedding.weight.data.copy_(pretrained)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(lstm_module.EMBEDDING_DIM)
    print("GloVe embeddings loaded successfully.")
except FileNotFoundError:
    print(f"[WARNING] GloVe file not found at '{GLOVE_PATH}'. Using random embeddings.")
 
optimizer = torch.optim.Adam(model.parameters(), lr=lstm_module.LR,
                             weight_decay=lstm_module.WEIGHT_DECAY)
criterion = torch.nn.BCEWithLogitsLoss()
 
model     = model.to(device)
criterion = criterion.to(device)
 
# ── Training ──────────────────────────────────────────────────────────────────
train_losses, test_losses = [], []
train_accs,   test_accs   = [], []
 
for epoch in range(1, EPOCHS_NLP + 1):
    print(f"\nEpoch {epoch}/{EPOCHS_NLP}")
    tr_loss, tr_acc = lstm_module.train_epoch(model, train_loader, optimizer, criterion, device)
    te_loss, te_acc = lstm_module.evaluate_epoch(model, test_loader, criterion, device)
    print(f"---> Train  Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f}")
    print(f"---> Test   Loss: {te_loss:.4f} | Acc: {te_acc:.4f}")
    train_losses.append(tr_loss);  test_losses.append(te_loss)
    train_accs.append(tr_acc);     test_accs.append(te_acc)
 
# ── Plots ─────────────────────────────────────────────────────────────────────
plot_training_curves(train_losses, test_losses, train_accs, test_accs)
plot_final_summary(train_losses, test_losses, train_accs, test_accs)
 

# ===========================================================================
# TASK 2: TIME-SERIES
# ===========================================================================
print("=" * 80)
print("TASK 2: TIME-SERIES - RESTAURANT REVENUE FORECASTING")

"""
main.py
=======
Entry point for the restaurant sales time series pipeline.
Run with:  python main.py

Runs in two phases:
  Phase 1 — Baseline: original approach with known issues
             (differencing + cumsum drift, auto-regressive forecast collapse,
              data gap shown as straight line)
  Phase 2 — Improved: redesigned pipeline that fixes all three issues

Adjust the CONFIG section to change hyperparameters for both phases.
"""

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH     = "3assign/data/RestaurantData.csv"
TARGET_COL    = "2to5"
LOOKBACK      = 30
HORIZON       = 30       # used only in improved pipeline
TRAIN_RATIO   = 0.80
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
FORECAST_DAYS = 365
SEED          = 42


def run_baseline():
    """
    Phase 1: original pipeline.
    Produces the flawed plots to motivate the redesign.
    """
    print("\n" + "="*60)
    print("  PHASE 1 — BASELINE (original approach)")
    print("  Watch for: drift in predictions, arch-shaped forecast,")
    print("  straight line across the data gap.")
    print("="*60)

    df = baseline.load_and_clean(DATA_PATH)

    # Stationarity + differencing (fed to model — this causes drift)
    stationary_series, d = baseline.make_stationary(df[TARGET_COL])
    baseline.plot_stationarity(df[TARGET_COL], stationary_series, df['DMY'])
    print(f"  Applied {d} round(s) of differencing.")

    values_stationary = stationary_series.values
    values_original   = df[TARGET_COL].values[-len(values_stationary):]

    scaled, scaler = baseline.scale_data(values_stationary)
    X, y           = baseline.create_sequences(scaled, LOOKBACK)
    X_train, X_test, y_train, y_test = baseline.split_data(X, y, TRAIN_RATIO)

    model  = baseline.LSTMForecaster(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    losses = baseline.train_model(model, X_train, y_train,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
    baseline.plot_training_loss(losses)

    # Inverse-difference via cumsum — this is where drift is introduced
    base_idx       = len(df[TARGET_COL]) - len(values_stationary)
    train_start    = base_idx + LOOKBACK
    test_start_abs = train_start + len(X_train)

    train_preds_diff = baseline.predict(model, X_train, scaler)
    test_preds_diff  = baseline.predict(model, X_test,  scaler)

    train_preds_orig = df[TARGET_COL].values[train_start - 1] + np.cumsum(train_preds_diff)
    test_preds_orig  = df[TARGET_COL].values[test_start_abs - 1] + np.cumsum(test_preds_diff)

    dates = df['DMY'].iloc[base_idx:].reset_index(drop=True)
    baseline.plot_predictions(dates, values_original, train_preds_orig,
                               test_preds_orig, LOOKBACK, len(X_train))

    # Auto-regressive forecast — this is where the arch collapse happens
    last_sequence = scaled[-LOOKBACK:, 0]
    future_diff   = baseline.forecast_future_autoregressive(model, last_sequence, scaler, FORECAST_DAYS)
    future_orig   = df[TARGET_COL].values[-1] + np.cumsum(future_diff)
    baseline.plot_forecast(df['DMY'], df[TARGET_COL].values, future_orig, FORECAST_DAYS)

    print("\n  Phase 1 complete — issues to note:")
    print("  1. Predictions drifted far above actual values (cumsum drift)")
    print("  2. Forecast collapsed into an arch shape (auto-regressive compounding)")
    print("  3. Straight line visible across the data gap (gap not handled)\n")


def run_improved():
    """
    Phase 2: redesigned pipeline with all three issues fixed.
    """
    print("\n" + "="*60)
    print("  PHASE 2 — IMPROVED PIPELINE")
    print("  Changes: train on original scale, direct multi-step")
    print("  forecasting, gap detection and handling.")
    print("="*60)

    df, segments = load_and_clean(DATA_PATH)
    print(f"  Full dataset: {len(df)} rows")

    test_stationarity(df[TARGET_COL], title="2to5")
    plot_stationarity(df, TARGET_COL)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[TARGET_COL].values.reshape(-1, 1))

    X, y = build_sequences_from_segments(segments, TARGET_COL, scaler, LOOKBACK, HORIZON)
    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_RATIO)
    print(f"  Sequences — train: {len(X_train)}, test: {len(X_test)}")

    model  = LSTMForecaster(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                             dropout=DROPOUT, horizon=HORIZON)
    losses = train_model(model, X_train, y_train,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
    plot_training_loss(losses)

    train_preds = predict(model, X_train, scaler)
    test_preds  = predict(model, X_test,  scaler)

    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_inv  = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    mae_train  = mean_absolute_error(y_train_inv[:, 0], train_preds[:, 0])
    rmse_train = mean_squared_error(y_train_inv[:, 0],  train_preds[:, 0]) ** 0.5
    mae_test   = mean_absolute_error(y_test_inv[:, 0],  test_preds[:, 0])
    rmse_test  = mean_squared_error(y_test_inv[:, 0],   test_preds[:, 0]) ** 0.5

    print(f"  Train set  |  MAE: {mae_train:.2f}  |  RMSE: {rmse_train:.2f}")
    print(f"  Test set   |  MAE: {mae_test:.2f}  |  RMSE: {rmse_test:.2f}")

    plot_predictions(df, TARGET_COL, train_preds, test_preds, LOOKBACK, len(X_train))

    future_preds, future_dates = forecast_future(
        model=model, df=df, target_col=TARGET_COL, scaler=scaler,
        lookback=LOOKBACK, horizon=HORIZON, forecast_days=FORECAST_DAYS,
    )
    plot_forecast(df, TARGET_COL, future_preds, future_dates)

    print("\n  Phase 2 complete.\n")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_baseline()
    run_improved()

    print("✓ Full pipeline complete.\n")


if __name__ == "__main__":
    main()

 
print("=" * 80)
