# data_prep.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config import TICKER, CSV_FILE

def download_historical(ticker=TICKER, period_days=365*2 + 20):
    """
    Download OHLCV for the ticker for ~2 years + buffer
    """
    end = datetime.today()
    start = end - timedelta(days=period_days)
    print(f"\n📊 Downloading {ticker} data from {start.date()} to {end.date()}...\n")
    
    df = yf.download(
    ticker,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    progress=True,
    auto_adjust=False  # ✅ keeps "Adj Close" column
    )

    
    if df.empty:
        raise RuntimeError("❌ Downloaded empty dataframe — check ticker symbol or internet connection.")
    
    df = df.sort_index()
    # standardize column names
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].rename(columns={'Adj Close': 'Adj_Close'})
    print(f"✅ Download complete: {len(df)} rows retrieved.")
    return df

def clean_data(df):
    """
    Cleans raw OHLCV data by filling small gaps and removing invalid rows
    """
    df = df.copy()
    df = df.ffill().bfill()  # fill missing values
    df = df[(df['Close'] > 0) & (df['Volume'] >= 0)]  # drop invalid
    return df

def main():
    try:
        df = download_historical()
        df_clean = clean_data(df)
        df_clean.to_csv(CSV_FILE)
        print(f"\n💾 Saved cleaned data to '{CSV_FILE}' ({len(df_clean)} rows).\n")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)
    main()
