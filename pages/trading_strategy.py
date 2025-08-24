# pages/trading_strategy.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📈 Trading Strategy Dashboard")

# -----------------------------
# Helper Functions
# -----------------------------
def load_data(tickers, period="1y", interval="1d"):
    return {ticker: yf.download(ticker, period=period, interval=interval, auto_adjust=False) for ticker in tickers}

def run_regression(df):
    df = df.reset_index()
    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]
    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])   # ✅ ensure slope is float
    trend = "Positive" if slope > 0 else "Negative"
    return model, trend, slope

def calculate_velocity(df):
    return df["Close"].pct_change(fill_method=None)

def calculate_bounce_efficiency(df):
    return df["Close"].pct_change(fill_method=None).abs().mean()

def find_best_ema(df, periods=[10, 20, 50, 100]):
    performance = {}
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
        signal = np.where(df["Close"] > df[f"EMA_{p}"], 1, -1)
        performance[p] = (signal * df["Close"].pct_change(fill_method=None)).cumsum().iloc[-1]
    best_period = max(performance, key=performance.get)
    return best_period, performance

def calculate_correlations(data, base="BTC-USD"):
    base_returns = data[base]["Close"].pct_change(fill_method=None)
    correlations = {}
    for ticker, df in data.items():
        if ticker == base:
            continue
        correlations[ticker] = base_returns.corr(df["Close"].pct_change(fill_method=None))
    best_coin = max(correlations, key=correlations.get)
    return best_coin, correlations

# -----------------------------
# Load Data
# -----------------------------
tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
data = load_data(tickers)

# -----------------------------
# BTC Trend Analysis
# -----------------------------
st.header("1️⃣ BTC Trend Analysis")
btc = data["BTC-USD"]
model, trend, slope = run_regression(btc)
st.write(f"BTC Trend: **{trend}** (Slope={slope:.6f})")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, model.predict(np.arange(len(btc)).reshape(-1,1)), label="Trend Line")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Best EMA for BTC
# -----------------------------
st.header("2️⃣ Best EMA for BTC")
best_ema, perf = find_best_ema(btc)
st.write(f"Best EMA period for BTC: **{best_ema}**")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, btc[f"EMA_{best_ema}"], label=f"EMA {best_ema}")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Best Coin vs BTC
# -----------------------------
st.header("3️⃣ Correlation with BTC")
best_coin, corrs = calculate_correlations(data)
st.write("Correlation values:", corrs)
st.write(f"✅ Best coin to trade in relation to BTC: **{best_coin}**")

# -----------------------------
# Apply EMA Strategy on Best Coin
# -----------------------------
st.header("4️⃣ Suggested Trades on Target Coin")
target = data[best_coin]
target[f"EMA_{best_ema}"] = target["Close"].ewm(span=best_ema, adjust=False).mean()
target["Signal"] = np.where(target["Close"] > target[f"EMA_{best_ema}"], 1, -1)

st.write(target[["Close", f"EMA_{best_ema}", "Signal"]].tail())

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(target.index, target["Close"], label=f"{best_coin} Price")
ax.plot(target.index, target[f"EMA_{best_ema}"], label=f"EMA {best_ema}")
buy_signals = target[target["Signal"] == 1]
sell_signals = target[target["Signal"] == -1]
ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="g", label="Buy")
ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="r", label="Sell")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Backtesting
# -----------------------------
st.header("5️⃣ Backtesting Strategy")
returns = target["Close"].pct_change(fill_method=None)
strategy_returns = target["Signal"].shift(1) * returns
cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(cumulative_returns.index, cumulative_returns, label="Strategy Equity Curve")
ax.legend()
st.pyplot(fig)

st.success("✅ Trading Strategy Completed!")




