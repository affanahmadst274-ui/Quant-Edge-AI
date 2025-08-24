import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📈 Trading Strategy")

# ---------------------------
# 1️⃣ Load Data
# ---------------------------
@st.cache_data
def load_data(tickers, start="2022-01-01"):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start)
        df.reset_index(inplace=True)

        # Normalize column names
        if "Adj Close" in df.columns:
            df.rename(columns={"Adj Close": "Close"}, inplace=True)

        data[ticker] = df[["Date", "Close"]].copy()
    return data

tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
data = load_data(tickers)

st.sidebar.header("Settings")
target_coin = st.sidebar.selectbox("Choose Target Coin", tickers, index=1)

# ---------------------------
# 2️⃣ Regression (BTC vs Coin)
# ---------------------------
def run_regression(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model

btc = data["BTC-USD"]["Close"].pct_change().dropna().values
coin = data[target_coin]["Close"].pct_change().dropna().values
length = min(len(btc), len(coin))
btc, coin = btc[:length], coin[:length]

model = run_regression(btc, coin)
sensitivity = model.coef_[0]

st.subheader("📊 Regression")
st.write(f"**Sensitivity of {target_coin} to BTC:** {sensitivity:.4f}")

fig, ax = plt.subplots()
ax.scatter(btc, coin, alpha=0.5)
ax.plot(btc, model.predict(btc.reshape(-1, 1)), color="red")
ax.set_xlabel("BTC Returns")
ax.set_ylabel(f"{target_coin} Returns")
st.pyplot(fig)

# ---------------------------
# 3️⃣ Find Best EMA
# ---------------------------
def find_best_ema(df, periods=[10, 20, 50, 100]):
    best_period, best_perf = None, -np.inf
    for p in periods:
        df["EMA"] = df["Close"].ewm(span=p, adjust=False).mean()
        df["Signal"] = np.where(df["Close"] > df["EMA"], 1, -1)
        df["Strategy"] = df["Signal"].shift(1) * df["Close"].pct_change()
        perf = df["Strategy"].cumsum().iloc[-1]
        if perf > best_perf:
            best_perf, best_period = perf, p
    return best_period, best_perf

best_period, _ = find_best_ema(data["BTC-USD"].copy())
st.subheader("🏆 Best EMA")
st.write(f"Best EMA for BTC: **{best_period}**")

# ---------------------------
# 4️⃣ Apply EMA & Suggest Trades
# ---------------------------
def suggest_trades(df, best_ema_period):
    df = df.copy()

    # Ensure "Close" column exists
    if "Close" not in df.columns:
        raise KeyError("DataFrame has no 'Close' column")

    df["EMA"] = df["Close"].ewm(span=best_ema_period, adjust=False).mean()
    df = df.dropna(subset=["Close", "EMA"])

    df["Signal"] = np.where(df["Close"] > df["EMA"], 1, -1)
    df["Position"] = df["Signal"].shift(1)
    df["Strategy"] = df["Position"] * df["Close"].pct_change()
    df["Cumulative"] = (1 + df["Strategy"]).cumprod()

    return df

st.header("4️⃣ Suggested Trades")
df_trades = suggest_trades(data[target_coin], best_period)
st.dataframe(df_trades[["Date", "Close", "EMA", "Signal", "Position", "Strategy"]].tail(20))

# ---------------------------
# 5️⃣ Plot Suggested Trades
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_trades["Date"], df_trades["Close"], label="Close Price")
ax.plot(df_trades["Date"], df_trades["EMA"], label=f"EMA {best_period}")
buy_signals = df_trades[df_trades["Signal"] == 1]
sell_signals = df_trades[df_trades["Signal"] == -1]
ax.scatter(buy_signals["Date"], buy_signals["Close"], marker="^", color="green", label="Buy", alpha=1)
ax.scatter(sell_signals["Date"], sell_signals["Close"], marker="v", color="red", label="Sell", alpha=1)
ax.legend()
st.pyplot(fig)



