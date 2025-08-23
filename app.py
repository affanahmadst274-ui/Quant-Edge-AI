import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

# =============================
# Helper Functions
# =============================

@st.cache_data
def load_financial_data_yf(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data[['Close']].rename(columns={'Close': 'close'})
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_correlation_and_sensitivity_relative_to_base(base_symbol, symbols, start_date, end_date):
    data = {}
    for sym in [base_symbol] + symbols:
        df = load_financial_data_yf(sym, start_date, end_date)
        if not df.empty:
            data[sym] = df['close']

    df_all = pd.DataFrame(data)
    if df_all.empty:
        return pd.DataFrame()

    corr_matrix = df_all.corr()

    results = []
    base_data = df_all[base_symbol]
    for sym in symbols:
        if sym in df_all:
            corr = corr_matrix.loc[base_symbol, sym]
            # Sensitivity using regression
            model = LinearRegression()
            model.fit(base_data.values.reshape(-1, 1), df_all[sym].values)
            sensitivity = model.coef_[0]
            results.append([sym, corr, sensitivity])
    return pd.DataFrame(results, columns=["Symbol", "Correlation", "Sensitivity"])

def predict_pair_value(base_symbol, target_symbol, base_value, start_date, end_date):
    base_data = load_financial_data_yf(base_symbol, start_date, end_date)
    target_data = load_financial_data_yf(target_symbol, start_date, end_date)

    if base_data.empty or target_data.empty:
        return None

    df = pd.DataFrame({
        'base': base_data['close'],
        'target': target_data['close']
    }).dropna()

    model = LinearRegression()
    model.fit(df[['base']], df['target'])
    predicted = model.predict(np.array([[base_value]]))[0]
    return float(predicted)

# =============================
# Streamlit Layout
# =============================

st.set_page_config(layout="wide")

# Sidebar Banner
st.sidebar.image("Pic1.PNG", use_container_width=True)

# Sidebar inputs
st.sidebar.header("📊 Data Parameters")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
base_symbol = st.sidebar.selectbox("Base Symbol", ["BTC-USD", "ETH-USD"])
symbols = st.sidebar.multiselect("Compare With", ["AVAX-USD", "SOL-USD", "BNB-USD", "ADA-USD"], default=["AVAX-USD"])

st.sidebar.header("🔮 Prediction Inputs")
target_symbol = st.sidebar.selectbox("Target Symbol", ["AVAXUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"])
base_value = st.sidebar.number_input("Base Value", value=50000)

# Top Banner
st.image("Pic2.PNG", use_container_width=True)

# Load data
crypto_data = {}
for sym in [base_symbol] + symbols:
    crypto_data[sym] = load_financial_data_yf(sym, start_date, end_date)

# Safely extract latest prices as floats
btc_price = float(crypto_data['BTC-USD']['close'].iloc[-1]) if not crypto_data['BTC-USD'].empty else 0.0
target_price_val = float(crypto_data[target_symbol.replace("USDT", "-USD")]['close'].iloc[-1]) if not crypto_data[target_symbol.replace("USDT", "-USD")].empty else 0.0

predicted_price = predict_pair_value(base_symbol, target_symbol.replace("USDT", "-USD"), base_value, start_date, end_date)

# =============================
# KPIs
# =============================
st.markdown("## 📈 Crypto Strength Analysis, Correlations and Predictions")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")
if predicted_price:
    kpi3.metric("Predicted Price", f"${predicted_price:,.2f}")
else:
    kpi3.metric("Predicted Price", "N/A")

st.markdown("---")

# =============================
# Main Content
# =============================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Correlation & Sensitivity Table")
    corr_df = calculate_correlation_and_sensitivity_relative_to_base(base_symbol, symbols, start_date, end_date)
    st.dataframe(corr_df)

    st.subheader("📉 Scatter Plots")
    for sym in symbols:
        if sym in crypto_data and not crypto_data[sym].empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crypto_data[base_symbol]['close'],
                y=crypto_data[sym]['close'],
                mode="markers",
                name=f"{base_symbol} vs {sym}"
            ))
            fig.update_layout(title=f"{base_symbol} vs {sym}", xaxis_title=base_symbol, yaxis_title=sym)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 BTC Candlestick")
    if not crypto_data['BTC-USD'].empty:
        fig = go.Figure(data=[go.Candlestick(
            x=crypto_data['BTC-USD'].index,
            open=crypto_data['BTC-USD']['close'],
            high=crypto_data['BTC-USD']['close'],
            low=crypto_data['BTC-USD']['close'],
            close=crypto_data['BTC-USD']['close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"📈 {target_symbol} Candlestick")
    target_key = target_symbol.replace("USDT", "-USD")
    if target_key in crypto_data and not crypto_data[target_key].empty:
        fig = go.Figure(data=[go.Candlestick(
            x=crypto_data[target_key].index,
            open=crypto_data[target_key]['close'],
            high=crypto_data[target_key]['close'],
            low=crypto_data[target_key]['close'],
            close=crypto_data[target_key]['close']
        )])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)


