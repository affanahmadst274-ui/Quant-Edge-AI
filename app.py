import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Crypto Dashboard")

# --------------------------------------------------
# Sidebar & Top Banners
# --------------------------------------------------
st.sidebar.image("Pic1.PNG", use_container_width=True)
st.image("Pic2.PNG", use_container_width=True)

# --------------------------------------------------
# Functions
# --------------------------------------------------
@st.cache_data
def download_crypto_data(symbols, period="6mo", interval="1d"):
    """Download crypto data from Yahoo Finance."""
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, period=period, interval=interval)
            if not df.empty:
                df.reset_index(inplace=True)
                df.rename(
                    columns={
                        "Date": "timestamp",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
                data[symbol] = df
        except Exception as e:
            st.warning(f"⚠️ Could not fetch {symbol}: {e}")
    return data


def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    """Calculate correlation and sensitivity of all symbols relative to a base symbol."""
    try:
        df = pd.DataFrame()
        for symbol, d in data.items():
            if not d.empty:
                df[symbol] = d["close"].pct_change()
        df.dropna(inplace=True)

        if base_symbol not in df.columns:
            return pd.DataFrame()

        results = []
        base_returns = df[base_symbol].values.reshape(-1, 1)

        for symbol in df.columns:
            if symbol == base_symbol:
                continue
            correlation = df[base_symbol].corr(df[symbol])
            model = LinearRegression().fit(base_returns, df[symbol].values)
            sensitivity = model.coef_[0]
            results.append(
                {"Symbol": symbol, "Correlation": correlation, "Sensitivity": sensitivity}
            )

        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error calculating correlation: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
available_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"]
selected_symbols = st.sidebar.multiselect(
    "Select cryptocurrencies", available_symbols, default=["BTC-USD", "ETH-USD"]
)
target_symbol = st.sidebar.selectbox("Target crypto", available_symbols, index=0)

# --------------------------------------------------
# Data Fetch
# --------------------------------------------------
crypto_data = download_crypto_data(selected_symbols, period="6mo", interval="1d")

# --------------------------------------------------
# KPIs
# --------------------------------------------------
st.markdown("## 📊 Key Metrics")
kpi1, kpi2, kpi3 = st.columns(3)

btc_price = (
    crypto_data["BTC-USD"]["close"].iloc[-1].item()
    if "BTC-USD" in crypto_data and not crypto_data["BTC-USD"].empty
    else 0.0
)

target_price_val = (
    crypto_data[target_symbol]["close"].iloc[-1].item()
    if target_symbol in crypto_data and not crypto_data[target_symbol].empty
    else 0.0
)

with kpi1:
    st.metric("BTC Price (USD)", f"${btc_price:,.2f}")

with kpi2:
    st.metric(f"{target_symbol} Price (USD)", f"${target_price_val:,.2f}")

with kpi3:
    st.metric("Tracked Coins", len(crypto_data))

# --------------------------------------------------
# Candlestick Chart
# --------------------------------------------------
st.markdown("## 📈 Price Chart")
if target_symbol in crypto_data and not crypto_data[target_symbol].empty:
    df = crypto_data[target_symbol]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    fig.update_layout(
        title=f"{target_symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"No data available for {target_symbol}")

# --------------------------------------------------
# Correlation & Sensitivity
# --------------------------------------------------
if len(selected_symbols) > 1 and "BTC-USD" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(
        crypto_data, "BTC-USD"
    )
    if not results_df.empty:
        st.markdown("### 📑 Correlation & Sensitivity to BTC")
        st.dataframe(results_df, use_container_width=True)

        # Optional Heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        df = pd.DataFrame()
        for symbol, d in crypto_data.items():
            if not d.empty:
                df[symbol] = d["close"].pct_change()
        df.dropna(inplace=True)

        if not df.empty:
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)





