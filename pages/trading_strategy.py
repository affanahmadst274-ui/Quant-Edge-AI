# step1_app.py
import streamlit as st
import pandas as pd
import yfinance as yf

# ============ DATA LOADER FUNCTION ============
def fetch_crypto_data(symbol: str, interval: str, days: int = 3) -> pd.DataFrame:
    interval = interval.lower()
    fetch_interval = {"4h": "60m"}.get(interval, interval)
    period = f"{days}d"

    df = yf.download(
        tickers=symbol,
        period=period,
        interval=fetch_interval,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Flatten multi-level columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    df = df.reset_index()

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    else:
        df.rename_axis("timestamp", inplace=True)
        df.reset_index(inplace=True)

    # Rename OHLCV
    df.rename(
        columns={
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Adj Close": "adj_close", "Volume": "volume"
        },
        inplace=True,
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    if interval == "4h" and fetch_interval == "60m":
        df = (
            df.set_index("timestamp")
              .resample("4H")
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
              .dropna()
              .reset_index()
        )

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close"])


# ============ STREAMLIT UI ============
st.set_page_config(page_title="Step 1 - Data Loader", layout="wide")

st.title("📊 Step 1 — Crypto Data Loader (Yahoo Finance)")

# Sidebar controls
st.sidebar.header("Data Settings")

symbol = st.sidebar.selectbox(
    "Select Crypto Symbol",
    ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD"],
    index=0,
)

interval = st.sidebar.selectbox(
    "Select Time Interval",
    ["1m", "5m", "15m", "30m", "60m", "4h", "1d", "1wk"],
    index=5,
)

days = st.sidebar.slider("Days of Historical Data", min_value=1, max_value=30, value=5)

# Fetch Data
if st.sidebar.button("Fetch Data"):
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try a different symbol or interval.")
    else:
        st.success(f"✅ Data fetched for {symbol} ({interval}, {days}d)")
        st.write("### Sample Data")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("### Data Summary")
        st.write(df.describe())

        st.line_chart(df.set_index("timestamp")["close"], use_container_width=True)
else:
    st.info("👈 Select parameters and click **Fetch Data**")





