import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📈 Trading Strategy Analysis")

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g. BTC-USD, ETH-USD)", "BTC-USD")
period = st.sidebar.selectbox("Select Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=4)
ema_periods = st.sidebar.multiselect("EMA Periods", [10, 20, 50, 100, 200], default=[20, 50, 200])

# --------------------------------------------------
# Download Data
# --------------------------------------------------
@st.cache_data(ttl=300, show_spinner=True)
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    return data

data = load_data(ticker, period, interval)

if data.empty:
    st.error("No data found for this ticker/period/interval.")
    st.stop()

# --------------------------------------------------
# EMA Strategy
# --------------------------------------------------
def calculate_ema_signals(data, ema_periods):
    result = {}
    for period in ema_periods:
        ema_series = data["Close"].ewm(span=period, adjust=False).mean()
        data[f"EMA_{period}"] = ema_series
        # Align indices explicitly to avoid ValueError
        aligned_close, aligned_ema = data["Close"].align(ema_series, join="inner")
        score = ((aligned_close > aligned_ema).mean()) * 100
        result[period] = score
    return result

ema_scores = calculate_ema_signals(data, ema_periods)

# --------------------------------------------------
# Linear Regression Trend
# --------------------------------------------------
def linear_regression_trend(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)
    return trend

data["Trend"] = linear_regression_trend(data)

# --------------------------------------------------
# Correlation Analysis
# --------------------------------------------------
def correlation_with_market(base_ticker, target_ticker, period="6mo", interval="1d"):
    base = yf.download(base_ticker, period=period, interval=interval, auto_adjust=True)
    target = yf.download(target_ticker, period=period, interval=interval, auto_adjust=True)
    if base.empty or target.empty:
        return None
    df = pd.DataFrame({
        "base": base["Close"].pct_change(fill_method=None),
        "target": target["Close"].pct_change(fill_method=None)
    }).dropna()
    return df.corr().iloc[0, 1]

corr_with_btc = correlation_with_market("BTC-USD", ticker)
corr_with_eth = correlation_with_market("ETH-USD", ticker)

# --------------------------------------------------
# Plots
# --------------------------------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"],
    name="Candlesticks"
))
for period in ema_periods:
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f"EMA_{period}"],
        mode="lines",
        name=f"EMA {period}"
    ))
fig.add_trace(go.Scatter(
    x=data.index, y=data["Trend"],
    mode="lines", name="Trend", line=dict(dash="dot", color="orange")
))

fig.update_layout(title=f"{ticker} Price Chart with EMA & Trend",
                  xaxis_rangeslider_visible=False,
                  height=600)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Results
# --------------------------------------------------
st.subheader("📊 Strategy Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### EMA Signals")
    for period, score in ema_scores.items():
        st.metric(label=f"Above EMA {period}", value=f"{score:.2f}%")

with col2:
    st.markdown("### Correlation with Market")
    if corr_with_btc is not None:
        st.metric(label="Correlation with BTC", value=f"{corr_with_btc:.2f}")
    if corr_with_eth is not None:
        st.metric(label="Correlation with ETH", value=f"{corr_with_eth:.2f}")

st.success("✅ Trading Strategy page loaded successfully.")

