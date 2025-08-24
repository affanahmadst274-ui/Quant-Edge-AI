import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Trading Strategy Page
# -----------------------------
st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📈 Trading Strategy Analysis")

# Sidebar inputs
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Enter Ticker (e.g., BTC-USD, ETH-USD):", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if df.empty:
    st.error("No data found for the selected ticker and date range.")
    st.stop()

# ✅ FIX: Ensure Close is always a Series
if isinstance(df["Close"], pd.DataFrame):
    df["Close"] = df["Close"].iloc[:, 0]

# Calculate returns
df["Return"] = df["Close"].pct_change(fill_method=None)

# -----------------------------
# EMA Strategy
# -----------------------------
st.subheader("📊 Exponential Moving Average (EMA) Strategy")

ema_periods = [20, 50, 100]
for period in ema_periods:
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

# Buy/Sell signals
df["Signal"] = np.where(df["Close"] > df["EMA_20"], 1, -1)

# Plot EMA strategy
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price", color="blue")
for period in ema_periods:
    ax.plot(df.index, df[f"EMA_{period}"], label=f"EMA {period}")
ax.legend()
ax.set_title(f"{ticker} EMA Strategy")
st.pyplot(fig)

# -----------------------------
# Regression Trend
# -----------------------------
st.subheader("📉 Trend Detection using Linear Regression")

df = df.dropna().copy()
X = np.arange(len(df)).reshape(-1, 1)
y = df["Close"].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price", color="blue")
ax.plot(df.index, trend_line, label="Trend Line", color="red", linestyle="--")
ax.legend()
ax.set_title(f"{ticker} Linear Regression Trend")
st.pyplot(fig)

# -----------------------------
# KPI Metrics
# -----------------------------
st.subheader("📌 Strategy Metrics")

col1, col2, col3 = st.columns(3)

# ✅ Convert Series to scalar for metrics
ema_scores = {
    period: (df["Close"] > df[f"EMA_{period}"]).mean() * 100
    for period in ema_periods
}

with col1:
    st.markdown("### EMA Signals")
    for period, score in ema_scores.items():
        st.metric(label=f"Above EMA {period}", value=f"{score:.2f}%")

# Correlation with Market
with col2:
    market = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    if not market.empty:
        # ✅ Ensure Close is Series
        if isinstance(market["Close"], pd.DataFrame):
            market["Close"] = market["Close"].iloc[:, 0]
        corr = df["Return"].corr(market["Close"].pct_change(fill_method=None))
        st.metric("Correlation with BTC", f"{corr:.2f}")

# Trend Direction
with col3:
    slope = model.coef_[0][0]
    trend = "📈 Uptrend" if slope > 0 else "📉 Downtrend"
    st.metric("Trend", trend)

# -----------------------------
# Final Recommendation
# -----------------------------
st.subheader("📢 Final Recommendation")

if slope > 0 and ema_scores[20] > 50:
    st.success("✅ Bullish signal: Consider Long positions")
elif slope < 0 and ema_scores[20] < 50:
    st.error("🚨 Bearish signal: Consider Short positions")
else:
    st.warning("⚠️ Neutral/Sideways market. Stay cautious.")





