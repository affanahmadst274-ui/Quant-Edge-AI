# trading_strategy.py

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Fetch Crypto Data
# -------------------------------
def fetch_crypto_data(symbol, interval, days):
    try:
        interval_map = {
            "1h": "60m",
            "4h": "60m",  # We'll resample to 4h later
            "1d": "1d",
        }
        fetch_interval = interval_map.get(interval, "60m")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval=fetch_interval)

        if df.empty:
            return pd.DataFrame()

        # Reset index to get timestamp column
        df.reset_index(inplace=True)

        # Rename Yahoo Finance columns to lowercase
        df.rename(
            columns={
                "Datetime": "timestamp",
                "Date": "timestamp",   # daily data uses Date
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # Keep only required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        # Resample to 4h if needed
        if interval == "4h" and fetch_interval == "60m":
            df.set_index("timestamp", inplace=True)
            df = df.resample("4H").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna().reset_index()

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()


# -------------------------------
# Step 2: Backtest Strategy
# -------------------------------
def backtest_strategy(df, initial_budget=5000):
    trades = []
    budget = initial_budget
    position = 0
    entry_price = 0

    # Example strategy: Moving Average Crossover
    df["SMA20"] = df["close"].rolling(20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()

    for i in range(1, len(df)):
        if df["SMA20"].iloc[i] > df["SMA50"].iloc[i] and position == 0:
            # Buy
            position = budget / df["close"].iloc[i]
            entry_price = df["close"].iloc[i]
            budget = 0
            trades.append(("BUY", df["timestamp"].iloc[i], entry_price))

        elif df["SMA20"].iloc[i] < df["SMA50"].iloc[i] and position > 0:
            # Sell
            budget = position * df["close"].iloc[i]
            trades.append(("SELL", df["timestamp"].iloc[i], df["close"].iloc[i]))
            position = 0

    # Final exit if holding
    if position > 0:
        budget = position * df["close"].iloc[-1]
        trades.append(("SELL", df["timestamp"].iloc[-1], df["close"].iloc[-1]))

    profit_loss = budget - initial_budget
    win_rate = 0
    if trades:
        wins = [1 for i in range(1, len(trades), 2)
                if trades[i][2] > trades[i - 1][2]]
        if len(trades) >= 2:
            win_rate = (sum(wins) / (len(trades) // 2)) * 100

    return {
        "trades": trades,
        "final_budget": budget,
        "pnl": profit_loss,
        "win_rate": win_rate,
    }


# -------------------------------
# Step 3: Plot Results
# -------------------------------
def plot_results(df, trades):
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["close"], label="Price", color="blue")

    for action, time, price in trades:
        if action == "BUY":
            plt.scatter(time, price, marker="^", color="green", label="Buy", alpha=1)
        else:
            plt.scatter(time, price, marker="v", color="red", label="Sell", alpha=1)

    plt.title("Trading Strategy Backtest")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Trading Strategy Backtest (Top 50 Coins)")

symbol = st.sidebar.text_input("Enter Symbol", "BTC-USD")
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"])
days = st.sidebar.number_input("Days of Data", min_value=10, max_value=365, value=90)

if st.sidebar.button("Fetch Data"):
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try a different symbol or interval.")
    else:
        st.success("✅ Data Fetched Successfully!")

        st.write("### Sample Data")
        st.dataframe(df.head())

        results = backtest_strategy(df)
        st.write("### 🔹 Backtest Results")
        st.write(f"Number of Trades: {len(results['trades'])}")
        st.write(f"Profit & Loss: ${results['pnl']:.2f}")
        st.write(f"Win %: {results['win_rate']:.2f}%")
        st.write(f"Ending Total Budget: ${results['final_budget']:.2f}")

        st.write("### 🔹 Sample Trades")
        trade_df = pd.DataFrame(results["trades"], columns=["Action", "Time", "Price"])
        st.dataframe(trade_df.head())

        plot_results(df, results["trades"])












