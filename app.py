import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

#---------------------------------#
# Page layout
st.set_page_config(page_title="Crypto Strength Analysis", layout="wide")

#---------------------------------#
# Sidebar
st.sidebar.image("crypto_banner.png")  # optional image at top

st.sidebar.header("Data Parameters")
interval_options = ['1d', '1h', '15m', '5m']
selected_interval = st.sidebar.selectbox("Select Interval", interval_options)

lookback_days = st.sidebar.slider("Select Lookback Period (Days)", 1, 30, 7)

# Define a list of crypto tickers
crypto_yf_tickers_list = ['BTC-USD','ETH-USD','SOL-USD','AVAX-USD','DOGE-USD','LTC-USD','DOT-USD']
target_symbol = st.sidebar.selectbox("Select Target Symbol", [t.replace("-USD","USDT") for t in crypto_yf_tickers_list if t != "BTC-USD"])

btc_price_input = st.sidebar.number_input("Enter BTC Price ($)", value=98000, step=500)

#---------------------------------#
# Data Loading Function
@st.cache_data
def load_financial_data_yf(ticker, interval, days):
    try:
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False, actions=False)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open','High','Low','Close']].reset_index()
        df.rename(columns={'Date':'timestamp','Open':'open','High':'high','Low':'low','Close':'close'}, inplace=True)
        return df
    except:
        return pd.DataFrame()

#---------------------------------#
# Correlation & Sensitivity
def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    if base_symbol not in data: return pd.DataFrame()
    base_df = data[base_symbol].set_index("timestamp")['close']
    results = []
    for pair, df in data.items():
        if pair == base_symbol: continue
        aligned = pd.concat([base_df, df.set_index("timestamp")['close']], axis=1).dropna()
        if aligned.empty: continue
        base_pct = aligned.iloc[:,0].pct_change().dropna()
        target_pct = aligned.iloc[:,1].pct_change().dropna()
        aligned_pct = pd.concat([base_pct,target_pct], axis=1).dropna()
        if len(aligned_pct) < 2: continue
        corr = aligned_pct.corr().iloc[0,1]
        X = aligned_pct.iloc[:,0].values.reshape(-1,1)
        y = aligned_pct.iloc[:,1].values
        reg = LinearRegression().fit(X,y)
        results.append({"Pair":pair.replace("-USD","USDT"), "Correlation Coefficient":corr, "Sensitivity (% Change)":reg.coef_[0]})
    return pd.DataFrame(results).sort_values(by="Correlation Coefficient", ascending=False).reset_index(drop=True)

#---------------------------------#
# Prediction
def predict_pair_value(data, base_symbol, target_symbol, target_base_price):
    if base_symbol not in data or target_symbol not in data: return None
    base_df = data[base_symbol].set_index("timestamp")['close']
    target_df = data[target_symbol].set_index("timestamp")['close']
    merged = pd.concat([base_df,target_df], axis=1).dropna()
    if merged.empty: return None
    X = merged.iloc[:,0].values.reshape(-1,1)
    y = merged.iloc[:,1].values.reshape(-1,1)
    if len(X)<2: return None
    model = LinearRegression().fit(X,y)
    return model.predict([[target_base_price]])[0][0]

#---------------------------------#
# Load Data
symbols_to_load = ['BTC-USD', target_symbol.replace("USDT","-USD")] + [s for s in crypto_yf_tickers_list if s not in ['BTC-USD', target_symbol.replace("USDT","-USD")]]
crypto_data = {s:load_financial_data_yf(s, selected_interval, lookback_days) for s in symbols_to_load}

#---------------------------------#
# Prices
btc_price = crypto_data['BTC-USD']['close'].iloc[-1] if not crypto_data['BTC-USD'].empty else 0
target_price_val = crypto_data[target_symbol.replace("USDT","-USD")]['close'].iloc[-1] if not crypto_data[target_symbol.replace("USDT","-USD")].empty else 0
predicted_price = predict_pair_value(crypto_data,'BTC-USD',target_symbol.replace("USDT","-USD"),btc_price_input)

#---------------------------------#
# Top KPIs
st.markdown("## Crypto Strength Analysis, Correlations and Predictions")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")
if predicted_price:
    kpi3.metric("Predicted Price", f"${predicted_price:,.2f}")

#---------------------------------#
# Correlation Table
st.subheader("Correlation and Sensitivity Data")
corr_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data,"BTC-USD")
if not corr_df.empty:
    st.dataframe(corr_df,use_container_width=True)

#---------------------------------#
# Charts Layout
chart1, chart2 = st.columns(2)

# BTC Candlestick
if not crypto_data['BTC-USD'].empty:
    df = crypto_data['BTC-USD']
    fig_btc = go.Figure(data=[go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    fig_btc.update_layout(title="BTC Candlestick Chart", xaxis_rangeslider_visible=False)
    chart1.plotly_chart(fig_btc, use_container_width=True)

# Target Candlestick
if not crypto_data[target_symbol.replace("USDT","-USD")].empty:
    df = crypto_data[target_symbol.replace("USDT","-USD")]
    fig_target = go.Figure(data=[go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
    )])
    fig_target.update_layout(title=f"{target_symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
    chart2.plotly_chart(fig_target, use_container_width=True)

#---------------------------------#
# Scatter Plot
st.subheader(f"Scatter Plot: BTC vs {target_symbol}")
if not crypto_data['BTC-USD'].empty and not crypto_data[target_symbol.replace("USDT","-USD")].empty:
    merged = pd.concat([
        crypto_data['BTC-USD'].set_index("timestamp")['close'],
        crypto_data[target_symbol.replace("USDT","-USD")].set_index("timestamp")['close']
    ], axis=1).dropna()
    st.scatter_chart(merged.rename(columns={"close":target_symbol}))
