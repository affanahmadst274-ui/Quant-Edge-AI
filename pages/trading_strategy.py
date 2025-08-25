# trading_strategy.py (Step 1 Streamlit App - FIXED)

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# Fetch crypto data function
# ----------------------------
def fetch_crypto_data(symbol, interval, days):
    try:
        df = yf.download(
            tickers=symbol,
            period=f"{days}d",
            interval=interval,
            progress=False
        )

        if df.empty:
            return pd.DataFrame()

        # Reset index to keep timestamp
        df.reset_index(inplace=True)

        # Rename columns consistently
        rename_map = {
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",  # fallback to "close"
            "Volume": "volume",
        }
        df.rename(columns=rename_map, inplace=True)

        # Ensure timestamp exists
        if "timestamp" not in df.columns:
            df["timestamp"] = df.index

        # Keep only available columns
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols]

        # Clean timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        return df

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Crypto Trading Strategy - Step 1", layout="wide")
st.title("📈 Crypto Trading Strategy - Step 1: Data Fetching")

st.sidebar.header("⚙️ Settings")

# User inputs
symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD)", value="BTC-USD")
interval = st.sidebar.selectbox(
    "Select Interval",
    options=["1m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo"],
    index=7,  # default = 1d
)
days = st.sidebar.slider("Select Number of Days", min_value=1, max_value=60, value=14)

# Fetch Data Button
if st.sidebar.button("Fetch Data"):
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try a different symbol or interval.")
    else:
        st.success(f"✅ Data fetched successfully for {symbol}")

        # Show dataframe preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(20))

        # Plot data (only if "close" exists)
        if "close" in df.columns:
            st.subheader("📈 Price Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["timestamp"], df["close"], label=f"{symbol} Price", color="blue")
            ax.set_title(f"{symbol} Closing Price")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠️ No 'close' column found to plot the price.")









