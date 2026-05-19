import yfinance as yf
import numpy as np
import pandas as pd
import torch

def fetch_data(symbol="GC=F", start="2010-01-01", end="2025-05-19"):
    # Ambil data lebih awal agar sinkronisasi tidak memotong terlalu banyak
    data_1d = yf.download(symbol, start=start, end=end, interval="1d")
    data_1w = yf.download(symbol, start=start, end=end, interval="1wk")
    data_1m = yf.download(symbol, start=start, end=end, interval="1mo")

    # Gunakan DataFrame utama (Daily) sebagai basis
    df = pd.DataFrame(index=data_1d.index)
    df['close_1d'] = data_1d['Close']
    df['open_1d'] = data_1d['Open']
    
    # SINKRONISASI: Gabungkan data Weekly dan Monthly ke index Daily
    # Menggunakan ffill() agar nilai weekly/monthly tersedia di setiap baris harian
    data_1w_resampled = data_1w['Close'].reindex(df.index, method='ffill')
    data_1m_resampled = data_1m['Close'].reindex(df.index, method='ffill')
    
    df['close_1w'] = data_1w_resampled
    df['close_1m'] = data_1m_resampled
    
    return df

def compute_bollinger(df, period=20, std_dev=2):
    prices = df['close_1d'].values
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        w = prices[i - period + 1 : i + 1]
        m = w.mean()
        s = w.std()
        mid[i] = m
        upper[i] = m + std_dev * s
        lower[i] = m - std_dev * s

    band = upper - lower
    pct_b = np.divide(prices - lower, band,
                      out=np.full(n, 0.5),
                      where=np.isfinite(band) & (band != 0))
    df['bb_pct'] = np.where(np.isfinite(pct_b), np.clip(pct_b, 0.0, 1.0), 0.5)
    df['bb_upper'] = upper
    df['bb_mid'] = mid
    df['bb_lower'] = lower
    return df

def create_dataset(df, horizon=1, upper_p=0.85, lower_p=0.15):
    # 1. Feature Engineering
    df['return_1d'] = df['close_1d'].pct_change()
    df['return_lag_1'] = df['return_1d'].shift(1)
    df['return_lag_2'] = df['return_1d'].shift(2)
    df['rolling_mean_return_5'] = df['return_1d'].rolling(window=5).mean()
    df['rolling_vol_5'] = df['return_1d'].rolling(window=5).std()
    
    # Log-ratios untuk cross-timeframe
    df['ratio_to_1w'] = np.log(df['close_1d'] / (df['close_1w'] + 1e-8))
    df['ratio_to_1m'] = np.log(df['close_1d'] / (df['close_1m'] + 1e-8))
    
    # BB %B centered at 0
    df['bb_pct_scaled'] = (df['bb_pct'] - 0.5) * 2.0

    # 2. Target Labeling (Forward-looking)
    df['future_return'] = df['close_1d'].pct_change(periods=horizon).shift(-horizon)

    # Batas dinamis menggunakan Quantile historis
    df['hist_rolling_return'] = df['close_1d'].pct_change(periods=horizon)
    df['upper_threshold'] = df['hist_rolling_return'].rolling(window=250).quantile(upper_p)
    df['lower_threshold'] = df['hist_rolling_return'].rolling(window=250).quantile(lower_p)

    def label_target(row):
        if pd.isna(row['future_return']) or pd.isna(row['upper_threshold']) or pd.isna(row['lower_threshold']):
            return np.nan
        
        if row['future_return'] > row['upper_threshold']:
            return 1.0  # Buy
        elif row['future_return'] < row['lower_threshold']:
            return 0.0  # Sell
        else:
            return 0.5  # Hold

    df['target'] = df.apply(label_target, axis=1)

    # 3. Cleanup & Final Tensors
    df_ready = df.dropna().copy()
    
    if len(df_ready) == 0:
        print("WARNING: Dataset kosong setelah dropna(). Periksa window size dan rentang data.")
        return torch.tensor([]), torch.tensor([]), df_ready

    features = ['bb_pct_scaled', 'return_lag_1', 'return_lag_2', 
                'rolling_mean_return_5', 'rolling_vol_5', 'ratio_to_1w', 'ratio_to_1m']

    X = torch.tensor(df_ready[features].values, dtype=torch.float32)
    Y = torch.tensor(df_ready['target'].values, dtype=torch.float32).reshape(-1, 1)

    return X, Y, df_ready

def split_dataset(X, Y, train_ratio=0.8):
    total_len = len(X)
    train_size = int(total_len * train_ratio)
    
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    return X_train, Y_train, X_test, Y_test
