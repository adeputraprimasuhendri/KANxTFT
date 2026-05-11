import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from kan import KAN
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy


# ==========================================
# PHASE 1: DATA ACQUISITION (YFINANCE)
# ==========================================
def download_mtf_data(ticker="BBCA.JK"):
    print(f"Downloading data for {ticker}...")

    # Download different resolutions
    # Note: 15m is limited to last 60 days in yfinance
    df_15m = yf.download(ticker, period="60d", interval="15m")
    df_1h = yf.download(ticker, period="60d", interval="60m")
    df_1d = yf.download(ticker, period="2y", interval="1d")

    # Clean multi-index columns and normalize timezones
    processed_dfs = []
    for df in [df_15m, df_1h, df_1d]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure index is timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        processed_dfs.append(df)

    return processed_dfs[0], processed_dfs[1], processed_dfs[2]


# ==========================================
# PHASE 2: FEATURE ENGINEERING & ALIGNMENT
# ==========================================
def prepare_features(df_15m, df_1h, df_1d):
    # 1. Base Features (15m)
    df = df_15m[['Close', 'High', 'Low', 'Volume']].copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # 2. Denoising (APR Proxy using Savitzky-Golay)
    df['smoothed_log'] = savgol_filter(df['log_return'].fillna(0), window_length=11, polyorder=3)

    # 3. ATR (Volatility for Labeling)
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift(1))
    low_cp = np.abs(df['Low'] - df['Close'].shift(1))
    df['atr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(20).mean()

    # 4. Multi-Timeframe Alignment (Preventing Leakage)
    # Align 1H data
    df_1h['log_return_h1'] = np.log(df_1h['Close'] / df_1h['Close'].shift(1))
    df = pd.merge_asof(df.sort_index(), df_1h[['log_return_h1']].sort_index(),
                       left_index=True, right_index=True, direction='backward')

    # Align 1D data (S&R Z-Score)
    # Simple S&R: Moving Average 20 days as dynamic S&R
    df_1d['sr_level'] = df_1d['Close'].rolling(20).mean()
    df_1d['sr_std'] = df_1d['Close'].rolling(20).std()
    df_1d['z_score_sr'] = (df_1d['Close'] - df_1d['sr_level']) / df_1d['sr_std']
    df_1d['log_return_d1'] = np.log(df_1d['Close'] / df_1d['Close'].shift(1))

    # Shift daily data by 1 to ensure we only use YESTERDAY'S daily data for today's 15m bars
    df_1d_shifted = df_1d[['z_score_sr', 'log_return_d1']].shift(1)

    df = pd.merge_asof(df.sort_index(), df_1d_shifted.sort_index(),
                       left_index=True, right_index=True, direction='backward')

    # 5. Triple Barrier Labeling (TBM)
    # Target: 1 (Profit), -1 (Loss), 0 (Time-out)
    tp_mult = 2.0
    sl_mult = 1.0
    window = 16  # bars

    df['target_val'] = 0
    for i in range(len(df) - window):
        price_start = df['Close'].iloc[i]
        vol = df['atr'].iloc[i]

        future_prices = df['Close'].iloc[i + 1: i + window]

        upside = price_start + (vol * tp_mult)
        downside = price_start - (vol * sl_mult)

        # Check barriers
        touched_up = future_prices[future_prices >= upside].index
        touched_down = future_prices[future_prices <= downside].index

        if len(touched_up) > 0 and (len(touched_down) == 0 or touched_up[0] < touched_down[0]):
            df.iloc[i, df.columns.get_loc('target_val')] = 1
        elif len(touched_down) > 0 and (len(touched_up) == 0 or touched_down[0] < touched_up[0]):
            df.iloc[i, df.columns.get_loc('target_val')] = -1

    df['target'] = df['target_val'] + 1  # Shift to 0, 1, 2 for CrossEntropy
    df['time_idx'] = np.arange(len(df))
    df['group'] = "ticker"

    return df.dropna()


# ==========================================
# PHASE 3: MODEL TRAINING (KAN & TFT)
# ==========================================
def train_models(df):
    features = ['log_return', 'smoothed_log', 'log_return_h1', 'log_return_d1', 'z_score_sr']
    X = df[features].values
    y = df['target'].values

    split = int(len(df) * 0.8)

    # --- KAN ---
    print("\n--- Training KAN (Spline-based) ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kan_ds = {
        'train_input': torch.tensor(X_scaled[:split], dtype=torch.float32),
        'test_input': torch.tensor(X_scaled[split:], dtype=torch.float32),
        'train_label': torch.tensor(y[:split], dtype=torch.long),
        'test_label': torch.tensor(y[split:], dtype=torch.long)
    }

    model_kan = KAN(width=[len(features), 8, 3], grid=3, k=3)
    model_kan.fit(kan_ds, opt="LBFGS", steps=10, loss_fn=nn.CrossEntropyLoss())

    # --- TFT ---
    print("\n--- Training TFT (Attention-based) ---")
    max_encoder_length = 24

    training_ds = TimeSeriesDataSet(
        df.iloc[:split],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=1,
        time_varying_unknown_reals=features,
        time_varying_known_reals=['time_idx']
    )

    train_loader = training_ds.to_dataloader(train=True, batch_size=32, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=4,
        loss=CrossEntropy()
    )

    trainer = Trainer(
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False
    )
    trainer.fit(tft, train_dataloaders=train_loader)

    # Save models and dataset parameters
    print("\nSaving models and dataset parameters...")
    import pickle
    import os
    os.makedirs("./model", exist_ok=True)
    
    model_kan.saveckpt(path="./model/kan_model.ckpt")
    trainer.save_checkpoint("./model/tft_model.ckpt")
    
    with open("./model/dataset_params.pkl", "wb") as f:
        pickle.dump(training_ds.get_parameters(), f)

    return model_kan, tft, training_ds


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    d15, d1h, dd = download_mtf_data("BBCA.JK")
    processed_df = prepare_features(d15, d1h, dd)
    print(f"Processed DataFrame size: {len(processed_df)}")
    kan_model, tft_model, _ = train_models(processed_df)

    print("\nTraining Complete. Models are ready for comparative inference.")