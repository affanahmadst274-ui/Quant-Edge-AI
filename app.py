import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# --------------------------------------------------
# Sidebar Config
# --------------------------------------------------
st.sidebar.title("⚙️ Settings")

# Refresh Interval
refresh_minutes = st.sidebar.slider("Auto-refresh interval (minutes)", 1, 30, 5)

# --------------------------------------------------
# Crypto Symbols
# --------------------------------------------------
symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "MATIC-USD", "APT-USD"]

target_symbol = st.sidebar.selectbox("Select a symbol", symbols, index=0)

# --------------------------------------------------
# Fetch Data Function
# --------------------------------------------------
@st.cache_data(ttl=300)
def get_data(symbol, interval="1d", period="1mo"):
    df = yf.download(symbol, period=period, interval=interval)
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
    return df

# --------------------------------------------------
# Price Prediction
# --------------------------------------------------
def predict_price(df):
    df = df.copy()
    df["time_index"] = np.arange(len(df))
    X = df[["time_index"]]
    y = df["close"]

    model = LinearRegression()
    model.fit(X, y)
    next_index = np.array([[len(df)]])
    predicted = model.predict(next_index)[0]
    return predicted

# --------------------------------------------------
# Layout
# --------------------------------------------------
st.title("📈 Crypto Price Prediction Dashboard")

col1, col2, col3 = st.columns(3)

# BTC Price
btc_data = get_data("BTC-USD")
btc_price = btc_data["close"].iloc[-1] if not btc_data.empty else 0
col1.metric("BTC Price", f"${btc_price:,.2f}")

# Target Coin Price
target_data = get_data(target_symbol)
target_price_val = target_data["close"].iloc[-1] if not target_data.empty else 0
col2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")

# Predictions (Auto + Manual unified)
auto_prediction = predict_price(target_data) if not target_data.empty else None
manual_prediction = None

with st.form("manual_predict_form"):
    manual_value = st.number_input("Enter your manual prediction", min_value=0.0, value=0.0)
    submitted = st.form_submit_button("Predict")
    if submitted and manual_value > 0:
        manual_prediction = manual_value

# Decide which prediction to show in KPI
if manual_prediction:
    col3.metric("Manual Predicted Price", f"${manual_prediction:,.2f}")
elif auto_prediction:
    col3.metric("Auto Predicted Price", f"${auto_prediction:,.2f}")
else:
    col3.metric("Predicted Price", "N/A")

# --------------------------------------------------
# Candlestick Chart
# --------------------------------------------------
if not target_data.empty:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=target_data["timestamp"],
                open=target_data["open"],
                high=target_data["high"],
                low=target_data["low"],
                close=target_data["close"],
            )
        ]
    )
    fig.update_layout(title=f"{target_symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Top Gainers & Losers
# --------------------------------------------------
st.subheader("📊 Market Movers (24h)")

market_data = {}
for sym in symbols:
    try:
        df = get_data(sym, period="2d")
        if len(df) >= 2:
            change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
            market_data[sym] = change
    except Exception as e:
        st.warning(f"{sym}: possibly delisted; no price data found")

if market_data:
    movers_df = pd.DataFrame(list(market_data.items()), columns=["Symbol", "Change (%)"])
    movers_df.sort_values("Change (%)", ascending=False, inplace=True)

    st.write("**Top Gainers**")
    st.dataframe(movers_df.head(5), use_container_width=True)

    st.write("**Top Losers**")
    st.dataframe(movers_df.tail(5), use_container_width=True)
else:
    st.write("No market data available.")
























