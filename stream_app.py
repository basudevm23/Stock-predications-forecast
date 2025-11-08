# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import plotly.graph_objs as go
# from features import add_features, sma, ema
# from config import MODEL_DIR
# from llm_gemini import ask_gemini_for_actions
# import datetime


# # ==============================================
# # ⚙️ Page Config
# # ==============================================
# st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")

# # ==============================================
# # 🧩 Helper Functions
# # ==============================================
# @st.cache_data
# def load_data(csv_path="hdfc_2y.csv"):
#     df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
#     df.columns = [c.replace(" ", "_") for c in df.columns]
#     return df

# @st.cache_data
# def generate_features(df):
#     df_feat = add_features(df)
#     df_feat["sma_50"] = sma(df_feat["Adj_Close"], 50)
#     df_feat["ema_50"] = ema(df_feat["Adj_Close"], 50)
#     df_feat["vol_50"] = df_feat["return"].rolling(50).std()
#     df_feat["return_lag_10"] = df_feat["return"].shift(10)
#     df_feat = df_feat.dropna()
#     return df_feat

# def prepare_target(df, horizon=5):
#     df[f'Adj_Close_next_{horizon}'] = df['Adj_Close'].shift(-horizon)
#     df['target'] = (df[f'Adj_Close_next_{horizon}'] > df['Adj_Close']).astype(int)
#     return df.dropna(subset=['target'])

# def load_selected_model(model_name):
#     if model_name == "Logistic Regression":
#         return joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))
#     elif model_name == "Random Forest":
#         return joblib.load(os.path.join(MODEL_DIR, "rf.joblib"))
#     elif model_name == "XGBoost":
#         return joblib.load(os.path.join(MODEL_DIR, "xgb.joblib"))
#     else:
#         raise ValueError("Unknown model name!")

# def run_predictions(df_feat, model):
#     feature_cols = [
#         'sma_10','sma_20','ema_12','ema_26','sma_50','ema_50',
#         'rsi_14','macd_line','macd_signal','macd_hist',
#         'bb_upper','bb_lower',
#         'vol_10','vol_20','vol_50','vol_ratio_1','price_sma20_ratio',
#         'return_lag_1','return_lag_2','return_lag_3','return_lag_4',
#         'return_lag_5','return_lag_10'
#     ]
#     X = df_feat[feature_cols]
#     preds = model.predict_proba(X)[:,1]
#     df_feat["Prediction"] = (preds > 0.5).astype(int)
#     df_feat["Probability"] = preds
#     return df_feat

# # ==============================================
# # 🎨 Streamlit UI
# # ==============================================
# st.title("📊 Stock Trend Prediction Dashboard")
# st.markdown("""
# Predict whether the stock price will go **Up (1)** or **Down (0)** in the next 5 days  
# using trained machine learning models (Logistic Regression, Random Forest, XGBoost).
# """)

# # Sidebar Config
# st.sidebar.header("⚙️ Configuration")
# csv_path = st.sidebar.text_input("CSV File Path", "hdfc_2y.csv")
# model_choice = st.sidebar.selectbox("Choose Model", ["XGBoost", "Random Forest", "Logistic Regression"])
# show_features = st.sidebar.checkbox("Show Feature DataFrame", False)

# # ==============================================
# # 🧮 Run Prediction
# # ==============================================
# try:
#     df_raw = load_data(csv_path)
#     df_feat = generate_features(df_raw)
#     df_feat = prepare_target(df_feat)

#     model = load_selected_model(model_choice)
#     df_pred = run_predictions(df_feat, model)

#     st.success(f"✅ {model_choice} predictions generated successfully!")

#     # ==============================================
#     # 📈 Chart
#     # ==============================================
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df_pred.index, y=df_pred["Adj_Close"],
#         mode='lines', name='Adjusted Close Price'
#     ))
#     fig.add_trace(go.Scatter(
#         x=df_pred[df_pred["Prediction"] == 1].index,
#         y=df_pred[df_pred["Prediction"] == 1]["Adj_Close"],
#         mode='markers', name='Predicted UP', marker=dict(color='green', size=6)
#     ))
#     fig.add_trace(go.Scatter(
#         x=df_pred[df_pred["Prediction"] == 0].index,
#         y=df_pred[df_pred["Prediction"] == 0]["Adj_Close"],
#         mode='markers', name='Predicted DOWN', marker=dict(color='red', size=6)
#     ))
#     fig.update_layout(
#         title=f"{model_choice} — Stock Price Prediction",
#         xaxis_title="Date", yaxis_title="Adjusted Close",
#         template="plotly_dark"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # ==============================================
#     # 📋 Display Predictions
#     # ==============================================
#     st.subheader("🔮 Recent Predictions")
#     st.dataframe(df_pred[["Adj_Close", "Prediction", "Probability"]].tail(15))

#     if show_features:
#         st.subheader("🧩 Feature DataFrame")
#         st.dataframe(df_feat.head())

#     # ==============================================
#     # 📥 Download Predictions
#     # ==============================================
#     csv = df_pred.to_csv().encode("utf-8")
#     st.download_button(
#         label="📥 Download Predictions CSV",
#         data=csv,
#         file_name=f"predictions_{model_choice}.csv",
#         mime="text/csv"
#     )

# except Exception as e:
#     st.error(f"❌ Error: {e}")

# # Footer
# st.markdown("---")
# st.markdown("📘 Developed by **Basudev Mohapatra** | Machine Learning Stock Prediction Project")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objs as go
from features import add_features, sma, ema
from config import MODEL_DIR
from llm_gemini import ask_gemini_for_actions
import datetime

# ==============================================
# ⚙️ Page Config
# ==============================================
st.set_page_config(page_title="📈 Stock Price Prediction Dashboard", layout="wide")

# ==============================================
# 🧩 Helper Functions
# ==============================================
@st.cache_data
def load_data(csv_path="hdfc_2y.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df

@st.cache_data
def generate_features(df):
    df_feat = add_features(df)
    df_feat["sma_50"] = sma(df_feat["Adj_Close"], 50)
    df_feat["ema_50"] = ema(df_feat["Adj_Close"], 50)
    df_feat["vol_50"] = df_feat["return"].rolling(50).std()
    df_feat["return_lag_10"] = df_feat["return"].shift(10)
    df_feat = df_feat.dropna()
    return df_feat

def prepare_target(df, horizon=5):
    df[f'Adj_Close_next_{horizon}'] = df['Adj_Close'].shift(-horizon)
    df['target'] = (df[f'Adj_Close_next_{horizon}'] > df['Adj_Close']).astype(int)
    return df.dropna(subset=['target'])

def load_selected_model(model_name):
    if model_name == "Logistic Regression":
        return joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))
    elif model_name == "Random Forest":
        return joblib.load(os.path.join(MODEL_DIR, "rf.joblib"))
    elif model_name == "XGBoost":
        return joblib.load(os.path.join(MODEL_DIR, "xgb.joblib"))
    else:
        raise ValueError("Unknown model name!")

def run_predictions(df_feat, model):
    feature_cols = [
        'sma_10','sma_20','ema_12','ema_26','sma_50','ema_50',
        'rsi_14','macd_line','macd_signal','macd_hist',
        'bb_upper','bb_lower',
        'vol_10','vol_20','vol_50','vol_ratio_1','price_sma20_ratio',
        'return_lag_1','return_lag_2','return_lag_3','return_lag_4',
        'return_lag_5','return_lag_10'
    ]
    X = df_feat[feature_cols]
    preds = model.predict_proba(X)[:,1]
    df_feat["Prediction"] = (preds > 0.5).astype(int)
    df_feat["Probability"] = preds
    return df_feat

# ==============================================
# 🎨 Streamlit UI
# ==============================================
st.title("📊 Stock Trend Prediction Dashboard")
st.markdown("""
Predict whether the stock price will go **Up (1)** or **Down (0)** in the next 5 days  
using trained machine learning models (Logistic Regression, Random Forest, XGBoost).
""")

# Sidebar Config
st.sidebar.header("⚙️ Configuration")
csv_path = st.sidebar.text_input("CSV File Path", "hdfc_2y.csv")
model_choice = st.sidebar.selectbox("Choose Model", ["XGBoost", "Random Forest", "Logistic Regression"])
show_features = st.sidebar.checkbox("Show Feature DataFrame", False)
use_llm = st.sidebar.checkbox("💡 Use Gemini LLM for Buy/Sell Suggestions", False)
llm_days = st.sidebar.slider("Analyze last N days", 5, 60, 15)

# ==============================================
# 🧮 Run Prediction
# ==============================================
try:
    df_raw = load_data(csv_path)
    df_feat = generate_features(df_raw)
    df_feat = prepare_target(df_feat)

    model = load_selected_model(model_choice)
    df_pred = run_predictions(df_feat, model)

    st.success(f"✅ {model_choice} predictions generated successfully!")

    # ==============================================
    # 📈 Chart
    # ==============================================
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred.index, y=df_pred["Adj_Close"],
        mode='lines', name='Adjusted Close Price'
    ))
    fig.add_trace(go.Scatter(
        x=df_pred[df_pred["Prediction"] == 1].index,
        y=df_pred[df_pred["Prediction"] == 1]["Adj_Close"],
        mode='markers', name='Predicted UP', marker=dict(color='green', size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df_pred[df_pred["Prediction"] == 0].index,
        y=df_pred[df_pred["Prediction"] == 0]["Adj_Close"],
        mode='markers', name='Predicted DOWN', marker=dict(color='red', size=6)
    ))
    fig.update_layout(
        title=f"{model_choice} — Stock Price Prediction",
        xaxis_title="Date", yaxis_title="Adjusted Close",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==============================================
    # 📋 Display Predictions
    # ==============================================
    st.subheader("🔮 Recent Predictions")
    st.dataframe(df_pred[["Adj_Close", "Prediction", "Probability"]].tail(15))

    if show_features:
        st.subheader("🧩 Feature DataFrame")
        st.dataframe(df_feat.head())

    # ==============================================
    # 🤖 Gemini LLM — Buy/Sell Recommendations
    # ==============================================
    if use_llm:
        st.subheader("🤖 Gemini Buy/Sell Recommendation")

        recent_df = df_pred.tail(llm_days)[["Adj_Close", "Prediction", "Probability"]]
        with st.spinner("💬 Analyzing with Gemini..."):
            prompt = (
                f"You are an expert financial analyst. Based on the last {llm_days} days of HDFC stock price data "
                f"and machine learning predictions, provide a short summary of market sentiment and suggest "
                f"a BUY, SELL, or HOLD action. Keep it under 150 words.\n\n"
                f"Data:\n{recent_df.to_string(index=True)}"
            )

            try:
                llm_response = ask_gemini_for_actions(prompt)
                st.markdown("### 🧠 Gemini Insight:")
                st.write(llm_response)
            except Exception as e:
                st.error(f"LLM call failed: {e}")

    # ==============================================
    # 📥 Download Predictions
    # ==============================================
    csv = df_pred.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name=f"predictions_{model_choice}.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("📘 Developed by **Basudev Mohapatra** | Machine Learning Stock Prediction Project")
