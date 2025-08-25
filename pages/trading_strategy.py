import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt

# ------------------------
# Fetch Crypto Data (Yahoo Finance)
# ------------------------
def fetch_crypto_data(symbol, interval="1h", days=7):
    try:
        period = f"{days}d"

        # yfinance interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        if interval == "1m" and days > 7:
            st.warning("⚠️ Yahoo Finance allows max 7 days of 1m data. Setting days=7.")
            period = "7d"

        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame()

        # Reset index and rename columns to match expected format
        df.reset_index(inplace=True)
        df.rename(
            columns={
                "Datetime": "timestamp",
                "Date": "timestamp",
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

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()


# ------------------------
# Streamlit UI
# ------------------------
st.title("📈 Trading Strategy Backtester")

symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD)", "BTC-USD")
interval = st.sidebar.selectbox(
    "Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
)
days = st.sidebar.slider("Days of Data", 1, 30, 7)

if st.sidebar.button("Fetch Data"):
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try another symbol or interval.")
    else:
        st.success(f"✅ Data fetched successfully! Rows: {len(df)}")
        st.write(df.head())

        st.line_chart(df.set_index("timestamp")["close"])










