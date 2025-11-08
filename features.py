# features.py
import pandas as pd
import numpy as np
import sys

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, lower

def add_features(df):
    df = df.copy()
    
    # ---- Ensure all numeric columns are proper floats ----
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Adj_Close'])  # remove rows with missing price

    # ---- Technical indicators ----
    df['return'] = df['Adj_Close'].pct_change(fill_method=None)
    df['sma_10'] = sma(df['Adj_Close'], 10)
    df['sma_20'] = sma(df['Adj_Close'], 20)
    df['ema_12'] = ema(df['Adj_Close'], 12)
    df['ema_26'] = ema(df['Adj_Close'], 26)
    df['rsi_14'] = rsi(df['Adj_Close'], 14)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df['Adj_Close'])
    df['bb_upper'], df['bb_lower'] = bollinger_bands(df['Adj_Close'])
    df['vol_10'] = df['return'].rolling(10).std()
    df['vol_20'] = df['return'].rolling(20).std()

    # ---- Lagged returns ----
    for lag in range(1, 6):
        df[f'return_lag_{lag}'] = df['return'].shift(lag)

    # ---- Volume-based features ----
    df['vol_avg_10'] = df['Volume'].rolling(10).mean()
    df['vol_ratio_1'] = df['Volume'] / (df['vol_avg_10'] + 1e-9)
    df['price_sma20_ratio'] = df['Adj_Close'] / (df['sma_20'] + 1e-9)

    df = df.dropna().copy()
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Usage: python features.py <csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    print(f"\n📈 Generating features for {input_csv} ...\n")

    # ---- Read CSV safely ----
    df = pd.read_csv(
        input_csv,
        index_col=0,
        parse_dates=[0],
        dayfirst=False,
        infer_datetime_format=True
    )

    df_feat = add_features(df)
    output_csv = "hdfc_2y_features.csv"
    df_feat.to_csv(output_csv)
    print(f"✅ Features generated and saved to '{output_csv}' ({len(df_feat)} rows).\n")
