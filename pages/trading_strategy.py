import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Trading Strategy", layout="wide")

# --------------------------
# Utility functions
# --------------------------

@st.cache_data
def load_data(symbols, start="2020-01-01"):
    data = {}
    for sym in symbols:
        df = yf.download(sym, start=start)
        data[sym] = df
    return data

def run_regression(df):
    df = df.dropna()
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    trend = "📈 Positive" if slope > 0 else "📉 Negative"
    return model, trend, float(slope)

def calculate_velocity(df):
    df["Returns"] = df["Close"].pct_change(fill_method=None)
    return df["Returns"].mean()

def calculate_bounce_efficiency(df):
    df["Volatility"] = df["Close"].pct_change(fill_method=None).rolling(10).std()
    return df["Volatility"].mean()

def find_best_ema(df, periods=[10, 20, 50, 100, 200]):
    best_score = -np.inf
    best_period = None
    close = df["Close"].dropna()

    for p in periods:
        ema = close.ewm(span=p, adjust=False).mean()
        # Align series
        aligned = pd.concat([close, ema], axis=1).dropna()
        score = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])  # close vs ema

        if score > best_score:
            best_score = score
            best_period = p

    return best_period, float(best_score)

def calculate_correlations(data, base="BTC-USD"):
    correlations = {}
    base_returns = data[base]["Close"].pct_change(fill_method=None).dropna()
    for sym, df in data.items():
        if sym == base:
            continue
        returns = df["Close"].pct_change(fill_method=None).dropna()
        aligned = pd.concat([base_returns, returns], axis=1).dropna()
        correlations[sym] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return correlations

def suggest_trades(df, best_ema_period):
    df = df.copy()
    df["EMA"] = df["Close"].ewm(span=best_ema_period, adjust=False).mean()

    # Ensure alignment before comparison
    df = df.dropna(subset=["Close", "EMA"])

    # Trading signals
    df["Signal"] = np.where(df["Close"] > df["EMA"], 1, -1)  # 1=Buy, -1=Sell

    # Position (shifted to avoid lookahead bias)
    df["Position"] = df["Signal"].shift(1).fillna(0)

    # Strategy Returns
    df["Returns"] = df["Close"].pct_change(fill_method=None)
    df["Strategy_Returns"] = df["Position"] * df["Returns"]

    return df

# --------------------------
# Streamlit App
# --------------------------

st.title("📊 Trading Strategy")

symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
data = load_data(symbols)

# 1️⃣ BTC Trend
st.header("1️⃣ BTC Trend Analysis")
btc = data["BTC-USD"]
model, trend, slope = run_regression(btc)
st.write(f"BTC Trend: **{trend}** (Slope={slope:.6f})")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, model.predict(np.arange(len(btc)).reshape(-1, 1)),
        label="Trend Line", linestyle="--")
ax.legend()
st.pyplot(fig)

# 2️⃣ Best EMA
st.header("2️⃣ BTC Best Performing EMA")
best_period, score = find_best_ema(btc)
st.write(f"Best EMA for BTC: **{best_period}** (Corr={score:.4f})")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(btc.index, btc["Close"], label="BTC Price")
ax.plot(btc.index, btc["Close"].ewm(span=best_period, adjust=False).mean(),
        label=f"EMA {best_period}")
ax.legend()
st.pyplot(fig)

# 3️⃣ Best Coin vs BTC
st.header("3️⃣ Best Coin in Relation to BTC")
correlations = calculate_correlations(data)
best_coin = max(correlations, key=correlations.get)
st.write("BTC Correlations:", correlations)
st.write(f"Best coin correlated with BTC: **{best_coin}** (Corr={correlations[best_coin]:.4f})")

# 4️⃣ Apply EMA & Suggest Trades
st.header("4️⃣ Suggested Trades")
df_trades = suggest_trades(data[best_coin], best_period)

# Show last 20 rows with signals
st.dataframe(df_trades[["Close", "EMA", "Signal", "Position", "Strategy_Returns"]].tail(20))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_trades.index, df_trades["Close"], label="Price")
ax.plot(df_trades.index, df_trades["EMA"], label=f"EMA {best_period}")

buy_signals = df_trades[df_trades["Signal"] == 1]
sell_signals = df_trades[df_trades["Signal"] == -1]
ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="g", label="Buy Signal")
ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="r", label="Sell Signal")

ax.legend()
st.pyplot(fig)

# 5️⃣ Backtesting Results
st.header("5️⃣ Backtesting")
cumulative_returns = (1 + df_trades["Returns"]).cumprod()
cumulative_strategy = (1 + df_trades["Strategy_Returns"]).cumprod()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cumulative_returns, label="Buy & Hold")
ax.plot(cumulative_strategy, label="Strategy")
ax.legend()
st.pyplot(fig)

st.success(f"📈 Final Strategy Return: {cumulative_strategy.iloc[-1]:.2f}x | Buy&Hold: {cumulative_returns.iloc[-1]:.2f}x")

