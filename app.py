import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# Sidebar banners
st.sidebar.image("Pic1.PNG", use_container_width=True)
st.image("Pic2.PNG", use_container_width=True)

# --------------------------------------------------
# Auto Refresh (simple version for all Streamlit versions)
# --------------------------------------------------
refresh_minutes = st.sidebar.slider("Auto-refresh interval (minutes)", 1, 30, 5)

# Use st.session_state counter to trigger rerun
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

st.session_state.refresh_counter += 1

# Inject JS to force page reload after interval
refresh_ms = refresh_minutes * 60 * 1000
st.markdown(
    f"""
    <meta http-equiv="refresh" content="{refresh_minutes * 60}">
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Tickers (Top 50 Coins by Market Cap - Yahoo format)
# --------------------------------------------------
TICKERS = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD","TRX-USD","DOT-USD","AVAX-USD",
    "MATIC-USD","LTC-USD","SHIB-USD","LINK-USD","BCH-USD","XLM-USD","ATOM-USD","UNI-USD","ETC-USD","XMR-USD",
    "TON-USD","HBAR-USD","APT-USD","ARB-USD","OP-USD","VET-USD","ICP-USD","NEAR-USD","FIL-USD","QNT-USD",
    "CRO-USD","AAVE-USD","GRT-USD","SAND-USD","MANA-USD","EOS-USD","FLOW-USD","EGLD-USD","XTZ-USD","CHZ-USD",
    "THETA-USD","AXS-USD","RUNE-USD","KAVA-USD","ZEC-USD","KSM-USD","DASH-USD","ENJ-USD","BAT-USD","ZIL-USD"
]

# --------------------------------------------------
# Data Fetching
# --------------------------------------------------
@st.cache_data(ttl=300)  # cache for 5 minutes
def load_data(tickers, period="6mo", interval="1d"):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not df.empty:
                df.reset_index(inplace=True)
                df.rename(columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume"
                }, inplace=True)
                data[ticker] = df
        except Exception as e:
            st.warning(f"⚠️ Could not load {ticker}: {e}")
    return data

crypto_data = load_data(TICKERS)

# --------------------------------------------------
# Sidebar Selection
# --------------------------------------------------
symbol = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_data.keys()))

# --------------------------------------------------
# Candlestick Chart
# --------------------------------------------------
st.subheader(f"{symbol} Price Chart")
df = crypto_data[symbol]
fig = go.Figure(data=[go.Candlestick(
    x=df["timestamp"],
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"]
)])
fig.update_layout(xaxis_rangeslider_visible=False, height=500)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation & Sensitivity
# --------------------------------------------------
st.subheader("Correlation & Sensitivity Analysis")

close_prices = pd.DataFrame({sym: crypto_data[sym]["close"] for sym in crypto_data.keys()})
close_prices.dropna(axis=1, inplace=True)

corr = close_prices.corr()
st.write("### Correlation Table")
st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

if "BTC-USD" in close_prices.columns:
    btc = close_prices["BTC-USD"].values.reshape(-1, 1)
    sensitivity = {}
    for coin in close_prices.columns:
        if coin == "BTC-USD":
            continue
        y = close_prices[coin].values
        model = LinearRegression().fit(btc, y)
        sensitivity[coin] = model.coef_[0]
    sens_df = pd.DataFrame.from_dict(sensitivity, orient="index", columns=["Sensitivity_to_BTC"]).sort_values(by="Sensitivity_to_BTC", ascending=False)
    st.write("### Sensitivity Table")
    st.dataframe(sens_df)

# --------------------------------------------------
# Top Gainers / Losers (last 24h)
# --------------------------------------------------
st.subheader("Top Gainers & Losers (24h)")

gains = {}
for sym, df in crypto_data.items():
    if len(df) > 1:
        change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
        gains[sym] = change

gains_df = pd.DataFrame(list(gains.items()), columns=["Symbol", "24h Change (%)"])
gains_df.sort_values(by="24h Change (%)", ascending=False, inplace=True)

col1, col2 = st.columns(2)
with col1:
    st.write("### Top 5 Gainers")
    st.dataframe(gains_df.head(5))
with col2:
    st.write("### Top 5 Losers")
    st.dataframe(gains_df.tail(5))
















