import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📈 Trading Strategy Analysis")

# Sidebar inputs
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Enter Crypto Symbol", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download data
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("No data found for the given ticker and date range.")
    st.stop()

# Calculate EMAs
ema_periods = [20, 50, 100]
for period in ema_periods:
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

# Buy/Sell signals using EMA_20
close_aligned, ema20_aligned = df["Close"].align(df["EMA_20"], join="inner")
df["Signal"] = np.where(close_aligned > ema20_aligned, 1, -1)

# Plot EMA strategy
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
for period in ema_periods:
    ax.plot(df.index, df[f"EMA_{period}"], label=f"EMA {period}", linestyle="--")
ax.set_title(f"{ticker} Price with EMA Strategy")
ax.legend()
st.pyplot(fig)

# Backtest returns
df["Market Return"] = df["Close"].pct_change(fill_method=None)
df["Strategy Return"] = df["Signal"].shift(1) * df["Market Return"]

cumulative_market = (1 + df["Market Return"]).cumprod()
cumulative_strategy = (1 + df["Strategy Return"]).cumprod()

# Plot cumulative returns
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cumulative_market, label="Market Return", color="blue")
ax.plot(cumulative_strategy, label="Strategy Return", color="green")
ax.set_title(f"{ticker} Backtest Results")
ax.legend()
st.pyplot(fig)

# Metrics
st.subheader("📊 Strategy Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Final Market Value", f"${cumulative_market.iloc[-1]:.2f}")
col2.metric("Final Strategy Value", f"${cumulative_strategy.iloc[-1]:.2f}")
col3.metric("Outperformance", f"{(cumulative_strategy.iloc[-1] - cumulative_market.iloc[-1]) * 100:.2f}%")

# EMA Scores (% of time above EMA)
ema_scores = {}
for period in ema_periods:
    close_aligned, ema_aligned = df["Close"].align(df[f"EMA_{period}"], join="inner")
    ema_scores[period] = (close_aligned > ema_aligned).mean() * 100

col1, col2 = st.columns(2)
with col1:
    st.markdown("### EMA Signals")
    for period, score in ema_scores.items():
        st.metric(label=f"Above EMA {period}", value=f"{score:.2f}%")

# Correlation with Market
with col2:
    st.markdown("### Correlation Matrix")
    corr = df[["Close"] + [f"EMA_{p}" for p in ema_periods]].corr()
    st.dataframe(corr)
