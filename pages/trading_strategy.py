import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy", layout="wide")

# --------------------------
# Utility Functions
# --------------------------
@st.cache_data
def load_data(ticker, start="2020-01-01"):
    df = yf.download(ticker, start=start)
    df = df[["Close"]]
    df.dropna(inplace=True)
    return df

def calculate_velocity(df):
    df["Velocity"] = df["Close"].pct_change()
    return df

def calculate_bounce_rate(df):
    df["Bounce"] = np.where(df["Velocity"] > 0, 1, -1)
    df["BounceRate"] = df["Bounce"].rolling(10).mean()
    return df

def find_best_ema(df, periods=[10, 20, 50, 100]):
    best_period = None
    best_score = -np.inf
    for p in periods:
        ema = df["Close"].ewm(span=p, adjust=False).mean()
        score = ((df["Close"] > ema).astype(int)).sum()  # how often price > EMA
        if score > best_score:
            best_score = score
            best_period = p
    return best_period

def correlation_with_btc(all_data, target="BTC-USD"):
    btc = all_data[target]["Close"]
    corrs = {}
    for coin, df in all_data.items():
        if coin != target:
            corrs[coin] = df["Close"].corr(btc)
    best_coin = max(corrs, key=corrs.get)
    return best_coin, corrs

def suggest_trades(df, best_ema_period):
    df = df.copy()
    # Ensure Close is Series
    df["EMA"] = df["Close"].ewm(span=best_ema_period, adjust=False).mean()
    df["Signal"] = np.where(df["Close"].values > df["EMA"].values, "Buy", "Sell")
    return df

# --------------------------
# Streamlit App Layout
# --------------------------
st.title("📈 Trading Strategy Module")

# 1️⃣ Load Data
st.header("1️⃣ Load Data")
coins = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
data = {c: load_data(c) for c in coins}
st.success("Data Loaded!")

# 2️⃣ BTC Trend & EMA
st.header("2️⃣ BTC Analysis")
btc = calculate_velocity(data["BTC-USD"])
btc = calculate_bounce_rate(btc)

best_period = find_best_ema(btc)
st.write(f"📊 Best EMA period for BTC: **{best_period}**")

fig, ax = plt.subplots(figsize=(10, 5))
btc["Close"].plot(ax=ax, label="BTC Close")
btc["Close"].ewm(span=best_period, adjust=False).mean().plot(ax=ax, label=f"EMA {best_period}")
plt.legend()
st.pyplot(fig)

# 3️⃣ Correlations
st.header("3️⃣ Correlation with BTC")
best_coin, corrs = correlation_with_btc(data)
st.write("🔗 Correlations:", corrs)
st.write(f"✅ Best coin to trade with BTC: **{best_coin}**")

# 4️⃣ Apply EMA & Suggest Trades
st.header("4️⃣ Suggested Trades")
df_trades = suggest_trades(data[best_coin], best_period)
st.dataframe(df_trades[["Close", "EMA", "Signal"]].tail(20))

fig, ax = plt.subplots(figsize=(10, 5))
df_trades["Close"].plot(ax=ax, label=f"{best_coin} Close")
df_trades["EMA"].plot(ax=ax, label=f"EMA {best_period}")
buy_signals = df_trades[df_trades["Signal"] == "Buy"]
sell_signals = df_trades[df_trades["Signal"] == "Sell"]
ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", label="Buy Signal")
ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", label="Sell Signal")
plt.legend()
st.pyplot(fig)

# 5️⃣ Backtesting (Simple)
st.header("5️⃣ Backtesting Results")
df_trades["Returns"] = df_trades["Close"].pct_change()
df_trades["Strategy"] = np.where(df_trades["Signal"] == "Buy", df_trades["Returns"], -df_trades["Returns"])
cumulative_strategy = (1 + df_trades["Strategy"].fillna(0)).cumprod()
cumulative_market = (1 + df_trades["Returns"].fillna(0)).cumprod()

fig, ax = plt.subplots(figsize=(10, 5))
cumulative_market.plot(ax=ax, label="Market Buy & Hold")
cumulative_strategy.plot(ax=ax, label="Strategy")
plt.legend()
st.pyplot(fig)







