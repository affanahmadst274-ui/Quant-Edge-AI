import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Trading Strategy", layout="wide")

# ---------------- Data Loader ----------------
@st.cache_data
def load_data(tickers, period="6mo", interval="1d"):
    data = {}
    for t in tickers:
        df = yf.download(t, period=period, interval=interval)
        data[t] = df
    return data

# ---------------- Regression ----------------
def run_regression(df):
    df = df.dropna()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    slope = float(model.coef_[0])  # ✅ ensure scalar
    trend = "📈 Positive" if slope > 0 else "📉 Negative"
    return model, trend, slope

# ---------------- Bounce Efficiency ----------------
def calculate_bounce_efficiency(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change(fill_method=None)
    df["Bounce"] = np.where(df["Return"] > 0, 1, 0)
    efficiency = 1 - df["Bounce"].mean()
    return efficiency

# ---------------- Best EMA Finder ----------------
def find_best_ema(df, periods=[10, 20, 50, 100, 200]):
    best_score = -np.inf
    best_period = None
    close = df["Close"]

    for p in periods:
        ema = close.ewm(span=p, adjust=False).mean()
        # ✅ Ensure aligned indexes
        ema, close_aligned = ema.align(close, join="inner")

        # Simple scoring: correlation between price and EMA
        score = close_aligned.corr(ema)

        if score > best_score:
            best_score = score
            best_period = p

    return best_period, best_score

# ---------------- Correlation ----------------
def correlation_with_btc(data, base="BTC-USD"):
    base_returns = data[base]["Close"].pct_change(fill_method=None)
    correlations = {}
    for t, df in data.items():
        if t == base: 
            continue
        returns = df["Close"].pct_change(fill_method=None)
        correlations[t] = base_returns.corr(returns)
    return correlations

# ---------------- Trade Suggestions ----------------
def suggest_trades(df, ema_period):
    df = df.copy()
    df["EMA"] = df["Close"].ewm(span=ema_period, adjust=False).mean()
    df["Signal"] = np.where(df["Close"] > df["EMA"], "BUY", "SELL")
    return df

# ---------------- Main ----------------
tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
data = load_data(tickers)

# 1️⃣ BTC Trend
st.header("1️⃣ BTC Trend Analysis")
btc = data["BTC-USD"]
model, trend, slope = run_regression(btc)
st.write(f"BTC Trend: **{trend}** (Slope={slope:.6f})")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.set_title("BTC Closing Price")
st.pyplot(fig)

# 2️⃣ BTC Best EMA
st.header("2️⃣ BTC Best Performing EMA")
best_period, score = find_best_ema(btc)
st.write(f"Best EMA for BTC: **{best_period}** (Corr={score:.4f})")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, btc["Close"].ewm(span=best_period, adjust=False).mean(),
        label=f"EMA {best_period}")
ax.legend()
st.pyplot(fig)

# 3️⃣ Best Coin vs BTC
st.header("3️⃣ Best Coin vs BTC (Correlation)")
correlations = correlation_with_btc(data)
best_coin = max(correlations, key=correlations.get)
st.write("Correlations:", correlations)
st.write(f"✅ Best Coin to trade vs BTC: **{best_coin}**")

# 4️⃣ Apply EMA to Target Coin
st.header("4️⃣ Applying BTC’s Best EMA to Target Coin")
target = data[best_coin]
target_with_signals = suggest_trades(target, best_period)
st.write(target_with_signals.tail())

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(target.index, target["Close"], label=f"{best_coin} Price")
ax.plot(target.index, target_with_signals["EMA"], label=f"EMA {best_period}")
ax.legend()
st.pyplot(fig)

# 5️⃣ Trade Suggestions
st.header("5️⃣ Trade Suggestions")
latest_signal = target_with_signals["Signal"].iloc[-1]
st.success(f"Suggested Trade for {best_coin}: **{latest_signal}**")





