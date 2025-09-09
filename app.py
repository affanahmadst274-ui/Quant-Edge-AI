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

    # Fix timestamp column for both daily & intraday
    if "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)

    df.rename(columns={
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
def train_model(base_data, target_data):
    df = pd.DataFrame({
        'base': base_data['close'] if isinstance(base_data['close'], pd.Series) else [base_data['close']],
        'target': target_data['close'] if isinstance(target_data['close'], pd.Series) else [target_data['close']]
    }).dropna()

    if len(df) < 2:
        return None, None

    X = df[['base']]
    y = df['target']

    model = LinearRegression()
    model.fit(X, y)
    return model, df

def predict_price(model, btc_input_price):
    if model is None:
        return None
    prediction = model.predict(np.array([[btc_input_price]]))
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

# Train model
model, df_pair = None, None
if "BTCUSDT" in crypto_data and target_symbol in crypto_data:
    model, df_pair = train_model(crypto_data["BTCUSDT"], crypto_data[target_symbol])

# Sidebar user input for prediction
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Manual Prediction")
btc_input_price = st.sidebar.number_input("Enter BTC Price (USD):", value=float(btc_price), step=100.0)

manual_triggered = st.sidebar.button("Predict")
manual_prediction = None

if manual_triggered:
    manual_prediction = predict_price(model, btc_input_price)

# Auto prediction (default latest BTC)
auto_prediction = None
if model is not None and df_pair is not None:
    latest_btc = df_pair['base'].iloc[-1]
    auto_prediction = predict_price(model, latest_btc)

# Show KPIs with distinction
kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")

if manual_prediction:
    kpi3.metric("Manual Predicted Price", f"${manual_prediction:,.2f}")
else:
    if auto_prediction:
        kpi3.metric("Auto Predicted Price", f"${auto_prediction:,.2f}")

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
# Correlation Heatmap (Removed)
# --------------------------------------------------
# st.markdown("### Correlation Matrix")
# if len(selected_symbols) > 1:
#     closes = pd.DataFrame({sym: crypto_data[sym]['close'] for sym in selected_symbols if not crypto_data[sym].empty})
#     corr = closes.corr()
#
#     heatmap = go.Figure(data=go.Heatmap(
#         z=corr.values,
#         x=corr.columns,
#         y=corr.index,
#         colorscale="RdBu",
#         zmin=-1,
#         zmax=1
#     ))
#     heatmap.update_layout(title="Crypto Correlation Heatmap")
#     st.plotly_chart(heatmap, use_container_width=True)


# --------------------------------------------------
# Correlation & Sensitivity to BTC (Removed)
# --------------------------------------------------
# def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
#     closes = pd.DataFrame({sym: data[sym]["close"] for sym in data if not data[sym].empty})
#     if closes.empty or base_symbol not in closes:
#         return pd.DataFrame()
#     
#     returns = closes.pct_change().dropna()
#
#     corr = returns.corr()[base_symbol]
#     sens = returns.corrwith(returns[base_symbol])
#
#     result = pd.DataFrame({
#         "Correlation to BTC": corr,
#         "Sensitivity to BTC": sens
#     })
#     return result
#
# if len(selected_symbols) > 1 and "BTCUSDT" in selected_symbols:
#     results_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data, "BTCUSDT")
#     if not results_df.empty:
#         st.markdown("### Correlation & Sensitivity to BTC")
#         st.dataframe(results_df.style.format("{:.2f}"), use_container_width=True)


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
        st.markdown("####  Top 5 Gainers")
        st.dataframe(movers_df.head(5).style.format({
            "Last Price (USD)": "${:,.2f}",
            "24h Change %": "{:.2f}%"
        }), use_container_width=True)

    with col2:
        st.markdown("####  Top 5 Losers")
        st.dataframe(movers_df.tail(5).style.format({
            "Last Price (USD)": "${:,.2f}",
            "24h Change %": "{:.2f}%"
        }), use_container_width=True)




# -------------------------------
# ARIMA Forecasting Model Section
# -------------------------------
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from itertools import product

warnings.filterwarnings("ignore")

st.markdown("## 📈 ARIMA Forecasting Model")

# Sidebar inputs for ARIMA
st.sidebar.subheader("ARIMA Model Parameters")

# ✅ Dropdown for ARIMA symbol (Top 50)
arima_symbol = st.sidebar.selectbox("Select Crypto for ARIMA", symbols, index=0)
crypto_symbol = arima_symbol.replace("USDT", "-USD")   # convert for Yahoo

prediction_ahead = st.sidebar.number_input(
    "Prediction Days Ahead", min_value=1, max_value=30, value=15, step=1
)

if st.sidebar.button("Run ARIMA Forecast"):
    # Step 1: Pull crypto data for the last 3 months
    btc_data = yf.download(crypto_symbol, period='3mo', interval='1d')
    btc_data = btc_data[['Close']].dropna()

    if len(btc_data) > 20:
        # Prepare train-test split (80% train, 20% test)
        train_size = int(len(btc_data) * 0.8)
        train, test = btc_data[:train_size], btc_data[train_size:]

        # Step 2: ARIMA model tuning
        p_values = range(0, 4)  
        d_values = range(0, 2)
        q_values = range(0, 4)

        def evaluate_arima_model(train, test, arima_order):
            try:
                model = ARIMA(train, order=arima_order)
                model_fit = model.fit()
                predictions = model_fit.forecast(steps=len(test))
                mse = mean_squared_error(test, predictions)
                return mse, model_fit
            except:
                return float('inf'), None

        results = []
        for p, d, q in product(p_values, d_values, q_values):
            arima_order = (p, d, q)
            mse, model_fit = evaluate_arima_model(train['Close'], test['Close'], arima_order)
            results.append((arima_order, mse, model_fit))

        # Select the best model
        best_order, best_mse, best_model = min(results, key=lambda x: x[1])
        forecast = best_model.forecast(steps=len(test) + prediction_ahead)

        # Latest close price and last predicted price
        latest_close_price = float(btc_data['Close'].iloc[-1])
        last_predicted_price = float(forecast[-1])

        # Show Metrics in Center
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-around;">
                    <div style="background-color: #d5f5d5; padding: 10px; border-radius: 10px; text-align: center;">
                        <h3>Latest Close Price</h3>
                        <p style="font-size: 20px;">${latest_close_price:,.2f}</p>
                    </div>
                    <div style="background-color: #d5f5d5; padding: 10px; border-radius: 10px; text-align: center;">
                        <h3>Price After {prediction_ahead} Days</h3>
                        <p style="font-size: 20px;">${last_predicted_price:,.2f}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

              # Plot ARIMA forecast (Simplified)
        plt.figure(figsize=(14, 5))

        # Actual price line
        plt.plot(btc_data.index, btc_data['Close'], label='Price', color='blue')

        # Forecast (future predictions)
        future_index = pd.date_range(start=btc_data.index[-1], periods=prediction_ahead + 1, freq='D')[1:]
        plt.plot(future_index, forecast[-prediction_ahead:], label=f'{prediction_ahead}-Day Forecast', color='red')

        plt.title(f'{crypto_symbol} ARIMA Model Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        st.pyplot(plt)





















