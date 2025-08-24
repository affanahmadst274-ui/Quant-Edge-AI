import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")

# --------------------------------------------------
# Sidebar: Auto-refresh
# --------------------------------------------------
refresh_minutes = st.sidebar.slider("Auto-refresh interval (minutes)", 1, 30, 5)
st.sidebar.markdown(f"⏳ Data refreshes every **{refresh_minutes} min**. Reload app manually if needed.")

# --------------------------------------------------
# Tickers (Top 50 Coins by Market Cap - Yahoo Finance format)
# --------------------------------------------------
top_50 = [
    "BTC-USD","ETH-USD","USDT-USD","BNB-USD","SOL-USD","XRP-USD","DOGE-USD","TON-USD","ADA-USD","AVAX-USD",
    "SHIB-USD","DOT-USD","TRX-USD","LINK-USD","MATIC-USD","LTC-USD","BCH-USD","XLM-USD","ATOM-USD","ETC-USD",
    "HBAR-USD","NEAR-USD","APT-USD","IMX-USD","ARB-USD","FIL-USD","CRO-USD","OP-USD","VET-USD","MNT-USD",
    "QNT-USD","GRT-USD","RNDR-USD","AAVE-USD","ALGO-USD","SAND-USD","MANA-USD","EGLD-USD","FLOW-USD","CHZ-USD",
    "AXS-USD","ICP-USD","KAVA-USD","NEO-USD","FTM-USD","RUNE-USD","ZIL-USD","1INCH-USD","ENJ-USD","XEC-USD"
]

base_symbol = "BTC-USD"
target_symbol = st.sidebar.selectbox("🎯 Select Target Crypto", [s for s in top_50 if s != base_symbol], index=1)

# --------------------------------------------------
# Fetch Data
# --------------------------------------------------
@st.cache_data(ttl=refresh_minutes*60)
def load_data(symbols, period="7d", interval="1h"):
    data = {}
    for sym in symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=period, interval=interval)
        if not hist.empty:
            hist.reset_index(inplace=True)
            hist.rename(columns={"Datetime": "timestamp"}, inplace=True)
            data[sym] = hist
    return data

crypto_data = load_data([base_symbol, target_symbol])

# --------------------------------------------------
# Latest Prices + Gainers/Losers
# --------------------------------------------------
latest_prices = {}
for sym in top_50:
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="1d", interval="1h")
        if not df.empty:
            latest_prices[sym] = {
                "price": df["Close"].iloc[-1],
                "change": (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
            }
    except Exception:
        pass

prices_df = pd.DataFrame(latest_prices).T.sort_values("change", ascending=False)
top_gainers = prices_df.head(5)
top_losers = prices_df.tail(5)

# --------------------------------------------------
# Layout
# --------------------------------------------------
st.title("📊 Crypto Price Prediction Dashboard")

# KPIs: Auto Prediction
col1, col2, col3 = st.columns(3)
btc_price = crypto_data[base_symbol]["Close"].iloc[-1] if base_symbol in crypto_data else None

if btc_price and target_symbol in crypto_data:
    df = pd.DataFrame({
        "base": crypto_data[base_symbol]["Close"],
        "target": crypto_data[target_symbol]["Close"]
    }).dropna()
    model = LinearRegression()
    model.fit(df[["base"]], df["target"])
    prediction = model.predict(np.array([[btc_price]]))[0]

    col1.metric("BTC Price (USD)", f"${btc_price:,.2f}")
    col2.metric(f"{target_symbol} Price", f"${crypto_data[target_symbol]['Close'].iloc[-1]:,.4f}")
    col3.metric(f"Predicted {target_symbol}", f"${prediction:,.4f}")
else:
    st.error("⚠️ Could not load enough data for prediction.")

# --------------------------------------------------
# Manual Prediction
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Manual Prediction Tool")

manual_btc_price = st.sidebar.number_input(
    "Enter custom BTC Price (USD)",
    min_value=1000.0,
    max_value=200000.0,
    value=float(btc_price) if btc_price else 20000.0,
    step=100.0
)

if st.sidebar.button("Predict Target Price"):
    if target_symbol in crypto_data and not crypto_data[target_symbol].empty:
        df = pd.DataFrame({
            "base": crypto_data[base_symbol]["Close"],
            "target": crypto_data[target_symbol]["Close"]
        }).dropna()
        if len(df) > 2:
            X, y = df[["base"]], df["target"]
            model = LinearRegression().fit(X, y)
            manual_prediction = model.predict(np.array([[manual_btc_price]]))[0]

            st.markdown("### 🎯 Manual Prediction Result")
            st.success(
                f"If **BTC = ${manual_btc_price:,.2f}**, "
                f"then **{target_symbol} ≈ ${manual_prediction:,.4f}**"
            )
        else:
            st.warning("Not enough data for manual prediction.")

# --------------------------------------------------
# Gainers & Losers
# --------------------------------------------------
col1, col2 = st.columns(2)
col1.subheader("🚀 Top 5 Gainers (24h)")
col1.table(top_gainers[["price", "change"]].style.format({"price": "${:,.4f}", "change": "{:+.2f}%"}))

col2.subheader("📉 Top 5 Losers (24h)")
col2.table(top_losers[["price", "change"]].style.format({"price": "${:,.4f}", "change": "{:+.2f}%"}))

# --------------------------------------------------
# Correlation & Sensitivity
# --------------------------------------------------
st.subheader("📈 Correlation & Sensitivity Analysis")

if len(crypto_data) > 1:
    closes = pd.DataFrame({sym: crypto_data[sym]["Close"] for sym in crypto_data if not crypto_data[sym].empty})
    corr = closes.corr()
    st.markdown("#### Correlation Table")
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None))

    # Sensitivity: linear regression slope
    st.markdown("#### Sensitivity Table (vs BTC)")
    sens = {}
    for sym in closes.columns:
        if sym != base_symbol:
            X, y = closes[[base_symbol]], closes[sym]
            model = LinearRegression().fit(X, y)
            sens[sym] = model.coef_[0]
    sens_df = pd.DataFrame.from_dict(sens, orient="index", columns=["Sensitivity"])
    st.dataframe(sens_df.style.format("{:.4f}"))

# --------------------------------------------------
# Charts
# --------------------------------------------------
st.subheader("📊 Charts")
if target_symbol in crypto_data:
    df = crypto_data[target_symbol]
    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )])
    fig.update_layout(title=f"{target_symbol} Candlestick", xaxis_title="Time", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)





















