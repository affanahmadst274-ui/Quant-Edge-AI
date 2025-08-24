import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# Sidebar banners
st.sidebar.image("Pic1.PNG", use_container_width=True)
st.image("Pic2.PNG", use_container_width=True)

# --------------------------------------------------
# Auto Refresh (default every 5 minutes)
# --------------------------------------------------
refresh_minutes = st.sidebar.slider("Auto-refresh interval (minutes)", 1, 30, 5)
st_autorefresh(interval=refresh_minutes * 60 * 1000, key="datarefresh")

# --------------------------------------------------
# Functions
# --------------------------------------------------
@st.cache_data
def load_crypto_data(symbol, period, interval):
    ticker = yf.Ticker(symbol.replace("USDT", "-USD"))
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Ensure timestamp exists
    if "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    else:
        df["timestamp"] = df.index

    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    return df


def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    if base_symbol not in data or data[base_symbol].empty:
        return pd.DataFrame()

    base_returns = data[base_symbol]["close"].pct_change().dropna()
    results = []

    for symbol, df in data.items():
        if symbol == base_symbol or df.empty:
            continue
        common_index = base_returns.index.intersection(df.index)
        if common_index.empty:
            continue
        returns = df.loc[common_index, "close"].pct_change().dropna()
        if returns.empty:
            continue

        corr = base_returns.loc[returns.index].corr(returns)
        X = base_returns.loc[returns.index].values.reshape(-1, 1)
        y = returns.values
        model = LinearRegression().fit(X, y)
        sensitivity = model.coef_[0]

        results.append({
            "Symbol": symbol,
            "Correlation": round(corr, 3),
            "Sensitivity": round(sensitivity, 3)
        })

    return pd.DataFrame(results)


def calculate_top_gainers_losers(data):
    changes = []
    for symbol, df in data.items():
        if df.empty:
            continue
        change = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        changes.append({"Symbol": symbol, "Change %": round(change, 2)})
    df_changes = pd.DataFrame(changes).sort_values("Change %", ascending=False)
    top_gainers = df_changes.head(5)
    top_losers = df_changes.tail(5)
    return top_gainers, top_losers

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.sidebar.header("Settings")

top_50_symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "DOTUSDT", "AVAXUSDT",
    "MATICUSDT", "LTCUSDT", "SHIBUSDT", "UNIUSDT", "LINKUSDT", "ATOMUSDT", "XLMUSDT", "ETCUSDT", "XMRUSDT", "TONUSDT",
    "BCHUSDT", "APTUSDT", "FILUSDT", "LDOUSDT", "ARBUSDT", "VETUSDT", "NEARUSDT", "OPUSDT", "QNTUSDT", "GRTUSDT",
    "AAVEUSDT", "SANDUSDT", "EGLDUSDT", "THETAUSDT", "ICPUSDT", "AXSUSDT", "MANAUSDT", "FLOWUSDT", "XTZUSDT", "RUNEUSDT",
    "NEOUSDT", "CAKEUSDT", "CHZUSDT", "ZILUSDT", "CRVUSDT", "1INCHUSDT", "ENSUSDT", "GMTUSDT", "SNXUSDT", "DYDXUSDT"
]

selected_symbols = st.sidebar.multiselect("Select cryptocurrencies", top_50_symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
days_back = st.sidebar.slider("Days of historical data", 30, 365, 180)
interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=6)

# --------------------------------------------------
# Data Loading
# --------------------------------------------------
crypto_data = {}
for sym in selected_symbols:
    crypto_data[sym] = load_crypto_data(sym, f"{days_back}d", interval)

# --------------------------------------------------
# Display Section
# --------------------------------------------------
st.title("📈 Crypto Price Prediction Dashboard")

# Show candlestick charts
for symbol in selected_symbols:
    df = crypto_data[symbol]
    if df.empty:
        st.warning(f"No data for {symbol}")
        continue

    st.subheader(f"{symbol} Candlestick Chart")
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation & Sensitivity Table
# --------------------------------------------------
if len(selected_symbols) > 1 and "BTCUSDT" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data, "BTCUSDT")
    if not results_df.empty:
        st.markdown("### 📊 Correlation & Sensitivity to BTC")
        st.dataframe(results_df, use_container_width=True)

# --------------------------------------------------
# Top Gainers & Losers
# --------------------------------------------------
top_gainers, top_losers = calculate_top_gainers_losers(crypto_data)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🚀 Top Gainers")
    st.dataframe(top_gainers, use_container_width=True)

with col2:
    st.markdown("### 📉 Top Losers")
    st.dataframe(top_losers, use_container_width=True)














