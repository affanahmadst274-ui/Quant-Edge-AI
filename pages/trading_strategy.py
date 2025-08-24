import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# Helper Functions
# ------------------------------

def load_data(tickers, start="2023-01-01"):
    data = {}
    for t in tickers:
        df = yf.download(t, start=start)
        df.dropna(inplace=True)
        data[t] = df
    return data

def run_regression(df):
    df = df.reset_index()
    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]
    model = LinearRegression().fit(X, y)
    trend = "Positive" if model.coef_[0] > 0 else "Negative"
    return model, trend, model.coef_[0]

def calculate_velocity(df):
    df["Velocity"] = df["Close"].pct_change(fill_method=None)
    return df

def calculate_bounce_rate_efficiency(df):
    # Bounce: how often price changes direction
    df["Diff"] = df["Close"].diff()
    df["Bounce"] = np.sign(df["Diff"]).diff().abs()
    efficiency = 1 - df["Bounce"].mean()
    return efficiency

def find_best_ema(df, periods=[10, 20, 50, 100, 200]):
    best_score = -np.inf
    best_period = None
    for p in periods:
        df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
        score = (df["Close"] > df[f"EMA_{p}"]).mean()
        if score > best_score:
            best_score = score
            best_period = p
    return best_period, best_score

def correlation_with_btc(data):
    btc = data["BTC-USD"]["Close"].pct_change(fill_method=None)
    corr = {}
    for coin, df in data.items():
        if coin != "BTC-USD":
            corr_val = df["Close"].pct_change(fill_method=None).corr(btc)
            corr[coin] = corr_val
    best_coin = max(corr, key=corr.get)
    return best_coin, corr

def suggest_trades(df, ema_period):
    df[f"EMA_{ema_period}"] = df["Close"].ewm(span=ema_period, adjust=False).mean()
    df["Signal"] = np.where(df["Close"] > df[f"EMA_{ema_period}"], 1, -1)
    return df

def backtest(df, initial_capital=1000):
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Close"].pct_change(fill_method=None)
    df["Equity"] = initial_capital * (1 + df["Strategy_Return"]).cumprod()
    return df

# ------------------------------
# Streamlit App
# ------------------------------

st.title("📊 Crypto Trading Strategy & Backtest")

tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
data = load_data(tickers)

# ---------------- Agenda Step 1: BTC Trend ----------------
st.header("1️⃣ BTC Trend Analysis")
btc = data["BTC-USD"]
model, trend, slope = run_regression(btc)
st.write(f"BTC Trend: **{trend}** (Slope={slope:.6f})")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, model.predict(np.arange(len(btc)).reshape(-1,1)), label="Trendline", linestyle="--")
ax.legend()
st.pyplot(fig)

# ---------------- Agenda Step 2: Velocity & Bounce ----------------
st.header("2️⃣ Velocity & Bounce Rate Efficiency")
btc = calculate_velocity(btc)
efficiency = calculate_bounce_rate_efficiency(btc)
st.write(f"Bounce Rate Efficiency: **{efficiency:.2f}**")

# ---------------- Agenda Step 3: Best EMA ----------------
st.header("3️⃣ Best EMA for BTC")
best_ema, score = find_best_ema(btc)
st.write(f"Best EMA Period: **{best_ema}** (Above EMA {score*100:.2f}% of the time)")

# ---------------- Agenda Step 4: Correlation ----------------
st.header("4️⃣ Correlation with BTC")
best_coin, corr = correlation_with_btc(data)
st.write("Correlations with BTC:")
st.json(corr)
st.write(f"Best Coin to Trade with BTC: **{best_coin}**")

# ---------------- Agenda Step 5: Apply Strategy ----------------
st.header("5️⃣ Suggested Trades")
target = data[best_coin].copy()
target = suggest_trades(target, best_ema)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(target.index, target["Close"], label=f"{best_coin} Price")
ax.plot(target.index, target[f"EMA_{best_ema}"], label=f"EMA {best_ema}")
ax.scatter(target.index, target["Close"], c=target["Signal"].map({1:"g",-1:"r"}), label="Buy/Sell", alpha=0.3)
ax.legend()
st.pyplot(fig)

# ---------------- Agenda Step 6: Backtest ----------------
st.header("6️⃣ Backtest Results")
bt = backtest(target)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(bt.index, bt["Equity"], label="Equity Curve")
ax.legend()
st.pyplot(fig)



