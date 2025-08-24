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

# Top 50 by market cap (USDT pairs mapped to Yahoo tickers as -USD)
symbols = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","TRXUSDT",
    "MATICUSDT","LTCUSDT","UNIUSDT","LINKUSDT","ATOMUSDT","ETCUSDT","XLMUSDT","IMXUSDT","APTUSDT","NEARUSDT",
    "OPUSDT","FILUSDT","ARBUSDT","VETUSDT","HBARUSDT","RNDRUSDT","INJUSDT","MKRUSDT","QNTUSDT","AAVEUSDT",
    "SANDUSDT","THETAUSDT","EOSUSDT","AXSUSDT","FLOWUSDT","CHZUSDT","XTZUSDT","MANAUSDT","KAVAUSDT","ZECUSDT",
    "RUNEUSDT","GRTUSDT","NEOUSDT","KSMUSDT","CRVUSDT","ENJUSDT","1INCHUSDT","DASHUSDT","ZILUSDT","COMPUSDT"
]

selected_symbols = st.sidebar.multiselect("Select Cryptos", symbols, default=["BTCUSDT", "ETHUSDT"])
target_symbol = st.sidebar.selectbox("Target Crypto", symbols, index=0)

days_back = st.sidebar.slider("Days of history", 30, 365, 180)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

# --------------------------------------------------
# Fetch Data
# --------------------------------------------------
@st.cache_data
def load_crypto_data(symbol, period, interval):
    ticker = yf.Ticker(symbol.replace("USDT", "-USD"))
    df = ticker.history(period=period, interval=interval)
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
for sym in symbols:  # Load all top 50 for movers
    crypto_data[sym] = load_crypto_data(sym, f"{days_back}d", interval)

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
# Correlation Heatmap
# --------------------------------------------------
st.markdown("### Correlation Matrix")

if len(selected_symbols) > 1:
    closes = pd.DataFrame({sym: crypto_data[sym]['close'] for sym in selected_symbols if not crypto_data[sym].empty})
    corr = closes.corr()

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

# --------------------------------------------------
# Correlation & Sensitivity Table
# --------------------------------------------------
def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    closes = pd.DataFrame({sym: data[sym]["close"] for sym in data if not data[sym].empty})
    if closes.empty or base_symbol not in closes:
        return pd.DataFrame()
    
    returns = closes.pct_change().dropna()

    corr = returns.corr()[base_symbol]
    sens = returns.corrwith(returns[base_symbol])

    result = pd.DataFrame({
        "Correlation to BTC": corr,
        "Sensitivity to BTC": sens
    })
    return result

if len(selected_symbols) > 1 and "BTCUSDT" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data, "BTCUSDT")
    if not results_df.empty:
        st.markdown("### Correlation & Sensitivity to BTC")
        st.dataframe(results_df.style.format("{:.2f}"), use_container_width=True)

# --------------------------------------------------
# Top Movers Section
# --------------------------------------------------
st.markdown("### 🚀 Top Movers (24h Change)")

def get_top_movers(data):
    movers = []
    for sym, df in data.items():
        if not df.empty and len(df) > 1:
            change = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
            movers.append((sym, df["close"].iloc[-1], change))
    return pd.DataFrame(movers, columns=["Symbol", "Last Price (USD)", "24h Change %"])

movers_df = get_top_movers(crypto_data)

if not movers_df.empty:
    movers_df = movers_df.sort_values("24h Change %", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🟢 Top 5 Gainers")
        st.dataframe(movers_df.head(5).style.format({
            "Last Price (USD)": "${:,.2f}",
            "24h Change %": "{:.2f}%"
        }), use_container_width=True)

    with col2:
        st.markdown("#### 🔴 Top 5 Losers")
        st.dataframe(movers_df.tail(5).style.format({
            "Last Price (USD)": "${:,.2f}",
            "24h Change %": "{:.2f}%"
        }), use_container_width=True)










