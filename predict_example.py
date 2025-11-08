# predict_example.py — Final Version (Aligned with model_train_eval.py)
import os
import pandas as pd
import numpy as np
import joblib
from features import add_features, sma, ema
from xgboost import XGBClassifier
from config import MODEL_DIR
from tensorflow.keras.models import load_model

def prepare_target(df, horizon=5):
    df[f'Adj_Close_next_{horizon}'] = df['Adj_Close'].shift(-horizon)
    df['target'] = (df[f'Adj_Close_next_{horizon}'] > df['Adj_Close']).astype(int)
    return df.dropna(subset=['target'])

def predict_from_csv(csv_path, model_name="XGBoost"):
    print(f"📥 Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df_feat = add_features(df)

    # === Add the same 4 extended features as used in model_train_eval.py ===
    df_feat["sma_50"] = sma(df_feat["Adj_Close"], 50)
    df_feat["ema_50"] = ema(df_feat["Adj_Close"], 50)
    df_feat["vol_50"] = df_feat["return"].rolling(50).std()
    df_feat["return_lag_10"] = df_feat["return"].shift(10)
    df_feat = df_feat.dropna()
    # =====================================================================

    df_feat = prepare_target(df_feat, horizon=5)

    feature_cols = [
        'sma_10','sma_20','ema_12','ema_26','sma_50','ema_50',
        'rsi_14','macd_line','macd_signal','macd_hist',
        'bb_upper','bb_lower',
        'vol_10','vol_20','vol_50','vol_ratio_1','price_sma20_ratio',
        'return_lag_1','return_lag_2','return_lag_3','return_lag_4',
        'return_lag_5','return_lag_10'
    ]

    X = df_feat[feature_cols]
    print(f"✅ Feature set ready: {X.shape}")

    # --- Load model ---
    model_path = os.path.join(MODEL_DIR, "xgb.joblib")

    if model_name == "LSTM":
        model_path = os.path.join(MODEL_DIR, "lstm_model.h5")

    print(f"🔍 Loading model: {model_path}")
    if model_name == "LSTM":
        model = load_model(model_path)
        X_scaled = (X - X.mean()) / (X.std() + 1e-9)
        WINDOW = 20
        X_seq = []
        for i in range(len(X_scaled) - WINDOW):
            X_seq.append(X_scaled.iloc[i:i+WINDOW].values)
        X_seq = np.array(X_seq)
        preds = model.predict(X_seq).ravel()
        y_pred = (preds > 0.5).astype(int)
        df_pred = df_feat.iloc[-len(y_pred):].copy()
        df_pred['LSTM_Pred'] = y_pred
        df_pred['LSTM_Prob'] = preds
    else:
        model = joblib.load(model_path)
        preds = model.predict_proba(X)[:,1]
        y_pred = (preds > 0.5).astype(int)
        df_pred = df_feat.copy()
        df_pred['Pred'] = y_pred
        df_pred['Prob'] = preds

    out_csv = f"predictions_{model_name}.csv"
    df_pred.to_csv(out_csv)
    print(f"💾 Predictions saved to {out_csv}")
    print(df_pred[['Adj_Close', 'Pred' if model_name!='LSTM' else 'LSTM_Pred']].tail(10))

if __name__ == "__main__":
    predict_from_csv("hdfc_2y.csv", model_name="XGBoost")
