# trading_strategy.py
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Strategy - Step 1", layout="wide")

# -------------------------
# Utility: fetch data
# -------------------------
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

    df = df.reset_index()

    # Standardize timestamp col
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    else:
        df.rename_axis("timestamp", inplace=True)
        df.reset_index(inplace=True)

    # Rename OHLCV cols to lowercase
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # Keep only useful cols
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[[c for c in cols if c in df.columns]].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Resample to 4h if needed
    if interval == "4h" and fetch_interval == "60m" and not df.empty:
        df = (
            df.set_index("timestamp")
              .resample("4H")
              .agg({
                  "open": "first",
                  "high": "max",
                  "low": "min",
                  "close": "last",
                  "volume": "sum"
              })
              .dropna()
              .reset_index()
        )

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# -------------------------
# Sidebar Controls (Step 1 UI)
# -------------------------
st.sidebar.header("📊 Step 1: Fetch Data")

# Top 50 coins (example, adjust as needed)
top_50 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD",
    "DOGE-USD", "TRX-USD", "MATIC-USD", "DOT-USD", "LTC-USD", "SHIB-USD",
    "AVAX-USD", "LINK-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "UNI-USD",
    "ICP-USD", "FIL-USD", "APT-USD", "ARB-USD", "QNT-USD", "NEAR-USD",
    "VET-USD", "HBAR-USD", "MKR-USD", "GRT-USD", "AAVE-USD", "ALGO-USD",
    "SAND-USD", "EGLD-USD", "THETA-USD", "XTZ-USD", "AXS-USD", "FLOW-USD",
    "CHZ-USD", "STX-USD", "IMX-USD", "KAVA-USD", "ZEC-USD", "EOS-USD",
    "RUNE-USD", "ENJ-USD", "BAT-USD", "CRV-USD", "NEO-USD", "1INCH-USD",
    "COMP-USD", "KSM-USD"
]

symbol = st.sidebar.selectbox("Select Symbol", top_50, index=0)
interval = st.sidebar.selectbox("Select Interval", ["1h", "4h", "1d"], index=1)
days = st.sidebar.number_input("Days of Data", min_value=1, max_value=90, value=7)

fetch_btn = st.sidebar.button("Fetch Data")

# -------------------------
# Main Content
# -------------------------
st.title("🚀 Crypto Trading Strategy - Step 1")

if fetch_btn:
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try different parameters.")
    else:
        st.success(f"✅ Data fetched for {symbol} | Interval: {interval} | Days: {days}")

        # Show table
        st.dataframe(df.tail(20), use_container_width=True)

        # Plot candlestick
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC"
        )])
        fig.update_layout(
            title=f"{symbol} - {interval.upper()} Candlestick",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)






