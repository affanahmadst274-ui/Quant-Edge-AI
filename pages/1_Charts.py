import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📊 Trading Strategy Analyzer")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Settings")
coin_list = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "MATIC-USD", "DOT-USD", "DOGE-USD", "LTC-USD"]
base_coin = st.sidebar.selectbox("Select Base Coin (Benchmark)", ["BTC-USD"])
target_coin = st.sidebar.selectbox("Select Target Coin", coin_list, index=1)
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)
ema_windows = st.sidebar.multiselect("EMA Windows", [5, 10, 20, 50, 100, 200], default=[10, 20, 50])

# --------------------------------------------------
# Data Fetching
# --------------------------------------------------
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

base_data = load_data(base_coin, period, interval)
target_data = load_data(target_coin, period, interval)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def calculate_ema_signals(data, ema_periods):
    result = {}
    for period in ema_periods:
        data[f"EMA_{period}"] = data["Close"].ewm(span=period, adjust=False).mean()
        score = ((data["Close"] > data[f"EMA_{period}"]).mean()) * 100
        result[period] = score
    return result

def regression_sensitivity(base, target):
    df = pd.DataFrame({
        "base": base["Close"].pct_change(),
        "target": target["Close"].pct_change()
    }).dropna()
    X = df["base"].values.reshape(-1,1)
    y = df["target"].values
    model = LinearRegression().fit(X,y)
    return model.coef_[0], model.intercept_, model.score(X,y)

# --------------------------------------------------
# Strategy Computation
# --------------------------------------------------
base_ema_scores = calculate_ema_signals(base_data.copy(), ema_windows)
best_base_ema = max(base_ema_scores, key=base_ema_scores.get)

sensitivity, intercept, r2 = regression_sensitivity(base_data, target_data)

target_ema_scores = calculate_ema_signals(target_data.copy(), [best_base_ema])

# --------------------------------------------------
# Layout
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📈 BTC Trend")
    st.metric("Best EMA (BTC)", best_base_ema, f"{base_ema_scores[best_base_ema]:.2f}% alignment")

with col2:
    st.subheader("🔗 Coin vs BTC")
    st.metric("Target Coin", target_coin, f"Sensitivity: {sensitivity:.2f}, R²={r2:.2f}")

with col3:
    st.subheader("📊 Strategy Signal")
    last_close = target_data["Close"].iloc[-1]
    last_ema = target_data[f"EMA_{best_base_ema}"].iloc[-1]
    signal = "LONG 🚀" if last_close > last_ema else "SHORT ⚡"
    st.metric("Signal", signal, f"Close={last_close:.2f}, EMA={last_ema:.2f}")

# --------------------------------------------------
# Charts
# --------------------------------------------------
st.markdown("### 📉 Charts")

tab1, tab2, tab3 = st.tabs(["Candlestick", "Regression", "EMA Scores"])

with tab1:
    fig = go.Figure(data=[go.Candlestick(
        x=target_data.index,
        open=target_data['Open'],
        high=target_data['High'],
        low=target_data['Low'],
        close=target_data['Close'],
        name="Candlestick"
    )])
    for ema in ema_windows:
        if f"EMA_{ema}" in target_data.columns:
            fig.add_trace(go.Scatter(x=target_data.index, y=target_data[f"EMA_{ema}"], mode="lines", name=f"EMA {ema}"))
    fig.update_layout(title=f"{target_coin} Candlestick with EMAs", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    df = pd.DataFrame({
        "base": base_data["Close"].pct_change(),
        "target": target_data["Close"].pct_change()
    }).dropna()
    X = df["base"].values.reshape(-1,1)
    y_pred = sensitivity * X + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["base"], y=df["target"], mode="markers", name="Data"))
    fig.add_trace(go.Scatter(x=df["base"], y=y_pred.flatten(), mode="lines", name="Regression"))
    fig.update_layout(title=f"{target_coin} Sensitivity to {base_coin}")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    score_df = pd.DataFrame.from_dict(base_ema_scores, orient="index", columns=["BTC Score"]).join(
        pd.DataFrame.from_dict(target_ema_scores, orient="index", columns=[f"{target_coin} Score"])
    )
    st.dataframe(score_df.style.background_gradient(cmap="Blues"), use_container_width=True)
