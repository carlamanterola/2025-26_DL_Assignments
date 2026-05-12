"""baseline_pipeline.py
====================
First-attempt pipeline — intentionally kept as originally implemented.
This is run before the improved pipeline so results can be compared.

Known issues (explained during presentation):
  - Model trained on differenced series + cumsum reconstruction → drift
  - Auto-regressive forecast (feeds own outputs back) → arch-shaped collapse
  - Data gap connected by a straight line (not handled)

These issues motivated the redesign in timeseries_pipeline.py.
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
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
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
    return df


# ─────────────────────────────────────────────
# STATIONARITY — ADF + DIFFERENCING
# ─────────────────────────────────────────────

def test_stationarity(series: pd.Series, title: str = "Series") -> bool:
    """
    Issue: differencing the series before training causes cumsum
    drift when reconstructing predictions back to original scale.
    """
    print(f"\n{'='*55}")
    print(f"  ADF Stationarity Test — {title}")
    print(f"{'='*55}")
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, _, _, critical_values, _ = result
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    for key, val in critical_values.items():
        print(f"  Critical {key}  : {val:.4f}")
    stationary = p_value < 0.05
    print(f"\n  Conclusion: {'STATIONARY ✓' if stationary else 'NON-STATIONARY ✗'}")
    print(f"{'='*55}\n")
    return stationary


def make_stationary(series: pd.Series) -> tuple:
    d = 0
    s = series.copy()
    while not test_stationarity(s, title=f"2to5 (d={d})"):
        s = s.diff().dropna()
        d += 1
        if d > 2:
            break
    return s, d


def plot_stationarity(original: pd.Series, stationary: pd.Series, dates: pd.Series):
    """
    Issue visible here: the straight line in the original series
    (top plot) is the data gap — not handled in this version.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    axes[0].plot(dates, original.values, color='steelblue', linewidth=0.8)
    axes[0].set_title("[BASELINE] Original Series — 2to5 Daily Sales", fontsize=13)
    axes[0].set_ylabel("Revenue")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    axes[1].plot(stationary.values, color='darkorange', linewidth=0.8)
    axes[1].set_title("Stationary Series (after differencing)", fontsize=13)
    axes[1].set_ylabel("Differenced Revenue")
    axes[1].set_xlabel("Time steps")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# DATA PREPARATION — SINGLE STEP
# ─────────────────────────────────────────────

def scale_data(series: np.ndarray) -> tuple:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler


def create_sequences(data: np.ndarray, lookback: int) -> tuple:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple:
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]


# ─────────────────────────────────────────────
# LSTM — SINGLE STEP OUTPUT
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """Single-step output — predicts only the next day."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0  = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def train_model(model, X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    loader    = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses    = []
    model.train()
    print("\n  Training baseline LSTM...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
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
# PREDICTIONS & FORECAST
# ─────────────────────────────────────────────

def predict(model, X, scaler):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).unsqueeze(-1)).numpy()
    return scaler.inverse_transform(preds).flatten()


def forecast_future_autoregressive(model, last_sequence, scaler, steps=365):
    """
    Issue: each prediction feeds back as input for the next step.
    Over 365 steps, errors compound and the forecast collapses
    into a smooth arch shape unrelated to actual sales patterns.
    """
    model.eval()
    sequence    = last_sequence.copy()
    future_preds = []
    with torch.no_grad():
        for _ in range(steps):
            x    = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(x).item()
            future_preds.append(pred)
            sequence = np.append(sequence[1:], pred)
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_training_loss(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='steelblue', linewidth=1)
    plt.title("[BASELINE] Training Loss per Epoch (MSE)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


def plot_predictions(dates, original, train_preds, test_preds, lookback, train_size):
    """
    Issue visible here: train predictions drift to ~38,000 due to
    cumsum reconstruction on top of differenced predictions.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, original, label="Actual", color='steelblue', linewidth=0.8, alpha=0.8)

    train_dates = dates.iloc[lookback: lookback + len(train_preds)]
    ax.plot(train_dates, train_preds, label="Train predictions", color='green', linewidth=1, alpha=0.85)

    test_start = lookback + train_size
    test_dates = dates.iloc[test_start: test_start + len(test_preds)]
    ax.plot(test_dates, test_preds, label="Test predictions", color='orange', linewidth=1, alpha=0.85)

    ax.set_title("[BASELINE] LSTM — In-Sample Predictions vs Actual", fontsize=13)
    ax.set_ylabel("Revenue (2–5pm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_forecast(dates, original, future_preds, forecast_days=365):
    """
    Issue visible here: arch-shaped forecast caused by auto-regressive
    error compounding over 365 recursive steps.
    """
    last_date    = dates.iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_days, freq='D')
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, original, label="Historical data", color='steelblue', linewidth=0.8, alpha=0.8)
    ax.plot(future_dates, future_preds, label="365-day forecast", color='crimson', linewidth=1.2)
    ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, label="Forecast start")
    ax.set_title("[BASELINE] LSTM — 365-Day Future Forecast", fontsize=13)
    ax.set_ylabel("Revenue (2–5pm)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()
