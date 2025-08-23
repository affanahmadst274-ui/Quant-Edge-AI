import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.image("Pic1.PNG", use_container_width=True)
st.sidebar.title("Crypto Dashboard")

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
selected_symbols = st.sidebar.multiselect("Select Cryptos", symbols, default=["BTCUSDT", "ETHUSDT"])
target_symbol = st.sidebar.selectbox("Target Crypto", symbols, index=0)
days_back = st.sidebar.slider("Days of history", 30, 365, 180)

# --------------------------------------------------
# Fetch Data
# --------------------------------------------------
@st.cache_data
def load_crypto_data(symbol, period):
    ticker = yf.Ticker(symbol.replace("USDT", "-USD"))
    df = ticker.history(period=period, interval="1d")
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.rename(columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)
    return df

crypto_data = {}
for sym in selected_symbols:
    crypto_data[sym] = load_crypto_data(sym, f"{days_back}d")

# --------------------------------------------------
# Prediction Model
# --------------------------------------------------
def predict_pair_value(base_data, target_data):
    df = pd.DataFrame({
        'base': base_data['close'] if isinstance(base_data['close'], pd.Series) else [base_data['close']],
        'target': target_data['close'] if isinstance(target_data['close'], pd.Series) else [target_data['close']]
    }).dropna()

    if len(df) < 2:
        return None

    X = df[['base']]
    y = df['target']

    model = LinearRegression()
    model.fit(X, y)

    latest_base_price = df['base'].iloc[-1]
    prediction = model.predict(np.array([[latest_base_price]]))
    return prediction[0]

# --------------------------------------------------
# Banners
# --------------------------------------------------
st.image("Pic2.PNG", use_container_width=True)

# --------------------------------------------------
# KPIs
# --------------------------------------------------
st.markdown("## Crypto Strength Analysis, Correlations and Predictions")
kpi1, kpi2, kpi3 = st.columns(3)

btc_price = crypto_data['BTCUSDT']['close'].iloc[-1].item() if not crypto_data['BTCUSDT'].empty else 0.0
target_price_val = crypto_data[target_symbol]['close'].iloc[-1].item() if not crypto_data[target_symbol].empty else 0.0

predicted_price = None
if "BTCUSDT" in crypto_data and target_symbol in crypto_data:
    predicted_price = predict_pair_value(crypto_data["BTCUSDT"], crypto_data[target_symbol])

kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")
if predicted_price:
    kpi3.metric("Predicted Price", f"${predicted_price:,.2f}")

# --------------------------------------------------
# Candlestick Charts
# --------------------------------------------------
st.markdown("### Price Charts (Candlestick)")

for symbol in selected_symbols:
    if symbol in crypto_data and not crypto_data[symbol].empty:
        df = crypto_data[symbol]
        fig = go.Figure(data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol
            )
        ])
        fig.update_layout(
            title=f"{symbol} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation + Sensitivity Tables
# --------------------------------------------------
st.markdown("### Correlation & Sensitivity Analysis")

if len(selected_symbols) > 1:
    closes = pd.DataFrame({sym: crypto_data[sym]['close'] for sym in selected_symbols if not crypto_data[sym].empty})

    # Correlation Table
    corr = closes.corr()
    st.subheader("Correlation Table")
    st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None))

    # Sensitivity (BTC as base)
    st.subheader("Sensitivity Table (vs BTC)")
    sensitivity = {}
    if "BTCUSDT" in closes.columns:
        for sym in closes.columns:
            if sym != "BTCUSDT":
                X = closes[["BTCUSDT"]].values
                y = closes[sym].values
                model = LinearRegression().fit(X, y)
                sensitivity[sym] = model.coef_[0]  # slope (sensitivity)

        sens_df = pd.DataFrame.from_dict(sensitivity, orient="index", columns=["Sensitivity to BTC"])
        st.dataframe(sens_df.style.background_gradient(cmap="Blues"))

    # Heatmap (keep both)
    heatmap = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    ))
    heatmap.update_layout(title="Crypto Correlation Heatmap")
    st.plotly_chart(heatmap, use_container_width=True)






