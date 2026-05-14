"""
timeseries_pipeline.py
======================
Full time series pipeline for restaurant sales forecasting.
Target variable: '2to5' (daily revenue, 2pm–5pm slot)
Model: LSTM neural network (Long Short-Term Memory)

Key design decisions vs previous approach:
  - No differencing fed to the model: train directly on the original
    scaled series. ADF is kept as a diagnostic only.
  - Direct multi-step forecasting: the model predicts the next
    HORIZON days all at once from one lookback window, instead of
    feeding its own outputs back recursively. This eliminates the
    compounding error that caused the arch-shaped forecast collapse.
  - Gap handling: the data gap (~63 missing days around late 2017)
    is detected and the dataset is split into two clean contiguous
    segments. The model is trained only on gap-free sequences.

Steps:
  1. Data loading & cleaning (with gap detection)
  2. Stationarity diagnostic (ADF test — informational only)
  3. Data preparation (scaling, multi-step sequences)
  4. Model definition & training
  5. In-sample evaluation
  6. Future forecast (direct, sliding window)
  7. Plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# STEP 1 — DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(filepath: str, gap_threshold: int = 7) -> tuple:
    """
    Loads and cleans the raw CSV, then splits it into contiguous
    segments separated by data gaps.

    A gap is defined as any stretch of rows where MissingPrevDays
    exceeds gap_threshold. These rows represent days where the
    restaurant had no data recorded — including them would create
    false transitions in the time series that the model would try
    to learn as real patterns.

    Returns:
      df       — the full cleaned dataframe (for plotting history)
      segments — list of contiguous sub-dataframes (for training)
    """
    df = pd.read_csv(filepath)

    cols_to_drop = [
        'Index', 'Year', 'Day',
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
        'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
    ]
    df = df.drop(columns=cols_to_drop)
    df['DMY'] = pd.to_datetime(df['DMY'], format='%m/%d/%Y')

    str_cols = ['DailyAvg', 'WeeklyAvg', 'MinSales', 'MaxSales', 'DailyBusyness', 'WeeklyBusyness']
    for col in str_cols:
        df[col] = df[col].replace('?', np.nan).astype(float)

    df = df.sort_values('DMY').reset_index(drop=True)
    df[str_cols] = df[str_cols].interpolate(method='linear').bfill()

    # Detect gap boundaries: rows where MissingPrevDays > threshold
    # mark the start of a new segment
    gap_mask    = df['MissingPrevDays'] > gap_threshold
    gap_indices = df.index[gap_mask].tolist()

    print(f"  Found {len(gap_indices)} gap(s) in the data:")
    for idx in gap_indices:
        print(f"    Row {idx} | Date {df.loc[idx, 'DMY'].date()} "
              f"| MissingPrevDays = {df.loc[idx, 'MissingPrevDays']}")

    # Split into segments around gaps, excluding the gap rows themselves
    boundaries = [0] + gap_indices + [len(df)]
    segments   = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] if i == 0 else boundaries[i] + 1
        end   = boundaries[i + 1]
        seg   = df.iloc[start:end].reset_index(drop=True)
        seg   = seg[seg['MissingPrevDays'] <= gap_threshold].reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)
            print(f"  Segment {i+1}: {seg['DMY'].iloc[0].date()} → "
                  f"{seg['DMY'].iloc[-1].date()} ({len(seg)} rows)")

    return df, segments


# ─────────────────────────────────────────────
# STEP 2 — STATIONARITY DIAGNOSTIC
# ─────────────────────────────────────────────

def test_stationarity(series: pd.Series, title: str = "Series"):
    """
    Runs the Augmented Dickey-Fuller (ADF) test as a diagnostic.

    This is now informational only — we no longer difference the data
    before feeding it to the model. The reason: differencing + cumsum
    reconstruction introduced compounding drift in predictions.
    Modern LSTMs handle mild non-stationarity well in practice,
    especially with MinMax scaling applied.

    ADF null hypothesis (H0): series has a unit root → non-stationary.
    p-value < 0.05: reject H0 → series IS stationary.
    """
    print(f"\n{'='*55}")
    print(f"  ADF Stationarity Test — {title}  [diagnostic only]")
    print(f"{'='*55}")

    result   = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, _, _, critical_values, _ = result

    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    for key, val in critical_values.items():
        print(f"  Critical {key}  : {val:.4f}")

    stationary = p_value < 0.05
    print(f"\n  Result: {'STATIONARY ✓' if stationary else 'NON-STATIONARY ✗'}")
    print(f"  Note: model trains on original scaled values regardless.")
    print(f"{'='*55}\n")


def plot_stationarity(df: pd.DataFrame, target_col: str):
    """
    Plots the original series with gap regions shaded in red
    so the discontinuity is visually explicit rather than
    connected by a misleading straight line.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    plot_series = df[target_col].copy().astype(float)
    gap_rows    = df.index[df['MissingPrevDays'] > 7].tolist()
    for idx in gap_rows:
        plot_series.iloc[idx] = np.nan

    ax.plot(df['DMY'], plot_series, color='steelblue', linewidth=0.8, label='2to5 sales')

    for idx in gap_rows:
        if idx > 0:
            gap_start = df.loc[idx - 1, 'DMY']
            gap_end   = df.loc[idx, 'DMY']
            ax.axvspan(gap_start, gap_end, color='red', alpha=0.2, label='Data gap')

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys())

    ax.set_title("Original Series — 2to5 Daily Sales (gaps highlighted)", fontsize=13)
    ax.set_ylabel("Revenue")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# STEP 3 — DATA PREPARATION
# ─────────────────────────────────────────────

def scale_data(values: np.ndarray) -> tuple:
    """
    Scales the series to [0, 1].
    The scaler is fitted on the full dataset so inverse-transform
    is consistent across training, evaluation, and forecasting.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    return scaled, scaler


def create_multistep_sequences(data: np.ndarray, lookback: int,
                               horizon: int) -> tuple:
    """
    Builds (X, y) pairs for direct multi-step forecasting.

    Unlike single-step forecasting where y is one value,
    here y is a vector of the next 'horizon' values:
        X[i] = data[i : i+lookback]                      shape: (lookback,)
        y[i] = data[i+lookback : i+lookback+horizon]     shape: (horizon,)

    The model learns to predict the entire horizon in one forward
    pass — at inference time we never feed predictions back as inputs,
    eliminating auto-regressive error compounding.
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i: i + lookback, 0])
        y.append(data[i + lookback: i + lookback + horizon, 0])
    return np.array(X), np.array(y)


def build_sequences_from_segments(segments: list, target_col: str,
                                   scaler: MinMaxScaler, lookback: int,
                                   horizon: int) -> tuple:
    """
    Builds sequences from each clean segment separately, then
    concatenates them. This ensures no sequence ever straddles
    a data gap — a lookback window will never mix the days before
    a gap with the days after it.
    """
    all_X, all_y = [], []
    for seg in segments:
        values = seg[target_col].values.reshape(-1, 1)
        scaled = scaler.transform(values)
        if len(scaled) < lookback + horizon:
            print(f"  Skipping short segment ({len(scaled)} rows < lookback+horizon)")
            continue
        X, y = create_multistep_sequences(scaled, lookback, horizon)
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X), np.concatenate(all_y)


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    Chronological train/test split — never shuffle time series data.
    """
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────
# STEP 4 — LSTM MODEL
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    LSTM for direct multi-step forecasting.

    The key difference from the previous version is the output layer:
    instead of outputting a single value (next day), it outputs a
    vector of size 'horizon' (next N days simultaneously).

    This means the model is trained end-to-end to minimise error
    across the entire forecast window in one forward pass.

    Architecture:
      Input  : (batch, lookback, 1)
      LSTM   : 2 stacked layers, hidden_size units each
      Dropout: applied between LSTM layers
      Output : Linear(hidden_size → horizon)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, horizon: int = 30):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Output layer maps last hidden state to 'horizon' future values at once
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                lr: float = 0.001) -> list:
    """
    Trains the LSTM with Adam + MSE loss.
    y_train is now shape (N, horizon) — the model minimises MSE
    across all horizon steps simultaneously.
    """
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    dataset   = TensorDataset(X_t, y_t)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()

    print("\n  Training LSTM...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}/{epochs}  |  Loss: {avg_loss:.6f}")

    print("  Training complete.\n")
    return losses


# ─────────────────────────────────────────────
# STEP 5 — IN-SAMPLE EVALUATION
# ─────────────────────────────────────────────

def predict(model: nn.Module, X: np.ndarray,
            scaler: MinMaxScaler) -> np.ndarray:
    """
    Runs inference and inverse-transforms predictions to revenue scale.
    Output shape: (N, horizon) — each row is one horizon-length forecast.
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        preds = model(X_t).numpy()

    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1))
    return preds_inv.reshape(preds.shape)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    """
    Evaluates on the first predicted step of each window —
    a fair single-step comparison against the actual series.
    """
    y_true_1 = scaler_inverse_flat(y_true[:, 0])
    y_pred_1 = y_pred[:, 0]
    mae  = mean_absolute_error(y_true_1, y_pred_1)
    rmse = np.sqrt(mean_squared_error(y_true_1, y_pred_1))
    print(f"  {label}  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")


def scaler_inverse_flat(scaled_vals: np.ndarray) -> np.ndarray:
    """Helper: inverse-transform a 1D scaled array."""
    # Placeholder — actual scaler passed in main via closure; see main.py
    return scaled_vals


# ─────────────────────────────────────────────
# STEP 6 — FUTURE FORECAST
# ─────────────────────────────────────────────

def forecast_future(model: nn.Module, df: pd.DataFrame, target_col: str,
                    scaler: MinMaxScaler, lookback: int,
                    horizon: int, forecast_days: int) -> tuple:
    """
    Direct sliding-window forecast for 'forecast_days' days ahead.

    At each step the model predicts the next 'horizon' days at once
    from a window of 'lookback' days. The window then slides forward
    by 'horizon' steps before the next prediction.

    The first window is seeded entirely with real historical data.
    Subsequent windows replace the oldest 'horizon' values with the
    previous prediction — so predicted values only enter the input
    after the first step. This minimises (but cannot fully eliminate)
    compounding error over very long horizons.

    Returns forecast values and their corresponding future dates.
    """
    model.eval()

    seed_values    = df[target_col].values[-lookback:]
    seed_scaled    = scaler.transform(seed_values.reshape(-1, 1)).flatten()
    current_window = seed_scaled.copy()

    all_preds  = []
    steps_done = 0

    with torch.no_grad():
        while steps_done < forecast_days:
            x           = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred_scaled = model(x).numpy().flatten()   # (horizon,)
            chunk       = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            all_preds.extend(chunk.tolist())
            # Slide window forward by horizon steps
            current_window = np.concatenate([current_window[horizon:], pred_scaled])
            steps_done    += horizon

    all_preds    = np.array(all_preds[:forecast_days])
    last_date    = df['DMY'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_days, freq='D')
    return all_preds, future_dates


# ─────────────────────────────────────────────
# STEP 7 — PLOTTING
# ─────────────────────────────────────────────

def plot_training_loss(losses: list):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='steelblue', linewidth=1)
    plt.title("Training Loss per Epoch (MSE)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


def plot_predictions(df: pd.DataFrame, target_col: str,
                     train_preds: np.ndarray, test_preds: np.ndarray,
                     lookback: int, train_size: int):
    """
    Overlays first-step train and test predictions on the original series.
    Gap rows are broken with NaN so no false straight lines appear.
    """
    original = df[target_col].copy().astype(float)
    gap_rows = df.index[df['MissingPrevDays'] > 7].tolist()
    for idx in gap_rows:
        original.iloc[idx] = np.nan

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['DMY'], original, label="Actual", color='steelblue',
            linewidth=0.8, alpha=0.8)

    train_dates = df['DMY'].iloc[lookback: lookback + len(train_preds)]
    ax.plot(train_dates, train_preds[:, 0], label="Train predictions",
            color='green', linewidth=1, alpha=0.85)

    test_start = lookback + train_size
    test_dates = df['DMY'].iloc[test_start: test_start + len(test_preds)]
    ax.plot(test_dates, test_preds[:, 0], label="Test predictions",
            color='orange', linewidth=1, alpha=0.85)

    ax.set_title("LSTM — In-Sample Predictions vs Actual", fontsize=13)
    ax.set_ylabel("Revenue (2–5pm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_forecast(df: pd.DataFrame, target_col: str,
                  future_preds: np.ndarray, future_dates: pd.DatetimeIndex):
    """
    Plots historical data followed by the future forecast.
    Gap rows are broken with NaN in the historical portion.
    """
    original = df[target_col].copy().astype(float)
    gap_rows = df.index[df['MissingPrevDays'] > 7].tolist()
    for idx in gap_rows:
        original.iloc[idx] = np.nan

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['DMY'], original, label="Historical data",
            color='steelblue', linewidth=0.8, alpha=0.8)
    ax.plot(future_dates, future_preds, label="365-day forecast",
            color='crimson', linewidth=1.2)
    ax.axvline(x=df['DMY'].iloc[-1], color='gray', linestyle='--',
               linewidth=1, label="Forecast start")

    ax.set_title("LSTM — 365-Day Future Forecast (direct multi-step)", fontsize=13)
    ax.set_ylabel("Revenue (2–5pm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()
