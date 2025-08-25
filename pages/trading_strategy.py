import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# -------------------------------
# Function to fetch crypto data
# -------------------------------
def fetch_crypto_data(symbol: str, interval: str, days: int = 30):
    try:
        ticker = yf.Ticker(symbol)

        # Map interval
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "1h",   # Yahoo doesn’t support 4h directly
            "1d": "1d"
        }

        fetch_interval = interval_map.get(interval, "1h")

        df = ticker.history(period=f"{days}d", interval=fetch_interval)

        if df.empty:
            return pd.DataFrame()

        # Reset index
        df.reset_index(inplace=True)

        # Rename columns to lowercase
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

        # For 4h aggregate manually
        if interval == "4h":
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
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crypto Data Viewer", layout="wide")

st.sidebar.title("⚡ Crypto Data Options")

# Top 50 coins (sample — you can expand later)
top_50 = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Solana (SOL)": "SOL-USD",
    "XRP": "XRP-USD",
    "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD",
    "Polygon (MATIC)": "MATIC-USD",
    "Litecoin (LTC)": "LTC-USD"
}

symbol_name = st.sidebar.selectbox("Select Coin", list(top_50.keys()))
symbol = top_50[symbol_name]

interval = st.sidebar.selectbox(
    "Select Interval",
    ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    index=5
)

days = st.sidebar.slider("Days of Data", 1, 90, 30)

# -------------------------------
# Fetch + Display Data (auto-fetch)
# -------------------------------
df = fetch_crypto_data(symbol, interval, days)

if df.empty:
    st.error("⚠️ No data available for this selection.")
else:
    st.success(f"✅ Data fetched for {symbol_name} | Interval: {interval}")

    # Show raw data
    st.subheader("📊 Price Data")
    st.dataframe(df.tail(20))

    # Candlestick chart
    st.subheader("📈 Candlestick Chart")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price"
            )
        ]
    )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)













