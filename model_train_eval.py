# model_train_eval.py  — tuned version
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config import MODEL_DIR, RANDOM_STATE
from features import add_features, sma, ema

os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_target(df, horizon=5):
    """Predict whether price will be higher after <horizon> days."""
    df = df.copy()
    df[f'Adj_Close_next_{horizon}'] = df['Adj_Close'].shift(-horizon)
    df['target'] = (df[f'Adj_Close_next_{horizon}'] > df['Adj_Close']).astype(int)
    return df.dropna(subset=['target'])

def time_split(df, frac=0.8):
    n = len(df)
    i = int(n*frac)
    return df.iloc[:i], df.iloc[i:]

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  "
          f"F1: {f1:.4f}  ROC_AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(f"{MODEL_DIR}/cm_{name}.png"); plt.close()

    fpr,tpr,_ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve: {name}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(f"{MODEL_DIR}/roc_{name}.png"); plt.close()

    return dict(name=name, accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)

def main():
    df = pd.read_csv("hdfc_2y.csv", index_col=0, parse_dates=True)
    df.columns = [c.replace(" ", "_") for c in df.columns]  # ✅ converts "Adj Close" → "Adj_Close"
    df_feat = add_features(df)

    # add slower indicators
    df_feat["sma_50"] = sma(df_feat["Adj_Close"], 50)
    df_feat["ema_50"] = ema(df_feat["Adj_Close"], 50)
    df_feat["vol_50"] = df_feat["return"].rolling(50).std()
    df_feat["return_lag_10"] = df_feat["return"].shift(10)
    df_feat = df_feat.dropna()

    df_tgt = prepare_target(df_feat, horizon=5)   # predict 5-day move

    feature_cols = [
        'sma_10','sma_20','ema_12','ema_26','sma_50','ema_50',
        'rsi_14','macd_line','macd_signal','macd_hist',
        'bb_upper','bb_lower',
        'vol_10','vol_20','vol_50','vol_ratio_1','price_sma20_ratio',
        'return_lag_1','return_lag_2','return_lag_3','return_lag_4',
        'return_lag_5','return_lag_10'
    ]

    X, y = df_tgt[feature_cols], df_tgt["target"]
    train, test = time_split(df_tgt)
    X_train, y_train = train[feature_cols], train["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    # logistic
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    joblib.dump(lr, f"{MODEL_DIR}/logreg.joblib")

    # random forest
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/rf.joblib")

    # tuned xgboost
    xgb = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_lambda=1.0, min_child_weight=3
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, f"{MODEL_DIR}/xgb.joblib")

    results=[]
    for model,name in [(lr,"LogisticRegression"),(rf,"RandomForest"),(xgb,"XGBoost")]:
        results.append(evaluate(model,X_test,y_test,name))

    pd.DataFrame(results).to_csv(f"{MODEL_DIR}/model_results_summary.csv",index=False)
    print("✅ Tuned models trained and saved in /models")

if __name__=="__main__":
    main()
