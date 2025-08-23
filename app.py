import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_data
def load_financial_data_yf(symbols, period="1y", interval="1d"):
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            if not hist.empty:
                data[symbol] = hist.rename(columns=str.lower)
        except Exception as e:
            st.warning(f"Could not fetch data for {symbol}: {e}")
    return data


def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol="BTC-USD"):
    results = []
    base = data.get(base_symbol)
    if base is None or base.empty:
        return pd.DataFrame()

    base_returns = base['close'].pct_change().dropna()

    for symbol, df in data.items():
        if symbol == base_symbol or df.empty:
            continue
        target_returns = df['close'].pct_change().dropna()
        aligned = pd.concat([base_returns, target_returns], axis=1).dropna()
        aligned.columns = ['base', 'target']

        correlation = aligned.corr().iloc[0, 1]
        model = LinearRegression().fit(aligned[['base']], aligned['target'])
        sensitivity = model.coef_[0]

        results.append({
            "Symbol": symbol,
            "Correlation": correlation,
            "Sensitivity to BTC": sensitivity
        })

    return pd.DataFrame(results)


def predict_pair_value(base_data, target_data):
    if base_data.empty or target_data.empty:
        return None

    # Ensure Series vs scalar handling
    df = pd.DataFrame({
        'base': base_data['close'] if isinstance(base_data['close'], pd.Series) else [base_data['close']],
        'target': target_data['close'] if isinstance(target_data['close'], pd.Series) else [target_data['close']]
    }).dropna()

    if df.empty:
        return None

    X = df[['base']]
    y = df['target']

    model = LinearRegression().fit(X, y)
    base_latest = df['base'].iloc[-1]
    predicted_value = model.predict([[base_latest]])[0]
    return predicted_value


# --------------------------------------------------
# Streamlit App Layout
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction App", layout="wide")

# Sidebar
st.sidebar.image("Pic1.PNG", use_column_width=True)
st.sidebar.title("Crypto Price Prediction App")
st.sidebar.markdown("Analyze crypto correlations, sensitivities, and predictions using Yahoo Finance data.")

# Top banner
st.image("Pic2.PNG", use_column_width=True)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.sidebar.header("Settings")
symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
selected_symbols = st.sidebar.multiselect("Select Cryptocurrencies", symbols, default=["BTC-USD", "ETH-USD"])
period = st.sidebar.selectbox("Period", ["1y", "6mo", "3mo", "1mo"])
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m"])

# --------------------------------------------------
# Load Data
# --------------------------------------------------
crypto_data = load_financial_data_yf(selected_symbols, period, interval)

# --------------------------------------------------
# Top KPIs
# --------------------------------------------------
st.markdown("## Crypto Strength Analysis, Correlations and Predictions")

kpi1, kpi2, kpi3 = st.columns(3)

btc_price = crypto_data['BTC-USD']['close'].iloc[-1].item() if 'BTC-USD' in crypto_data and not crypto_data['BTC-USD'].empty else 0.0

target_symbol = selected_symbols[1] if len(selected_symbols) > 1 else "ETH-USD"
target_price_val = crypto_data[target_symbol]['close'].iloc[-1].item() if target_symbol in crypto_data and not crypto_data[target_symbol].empty else 0.0

predicted_price = None
if "BTC-USD" in crypto_data and target_symbol in crypto_data:
    predicted_price = predict_pair_value(crypto_data['BTC-USD'], crypto_data[target_symbol])

kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")
if predicted_price:
    kpi3.metric("Predicted Price", f"${predicted_price:,.2f}")

# --------------------------------------------------
# Charts
# --------------------------------------------------
st.markdown("### Price Charts")
for symbol in selected_symbols:
    if symbol in crypto_data and not crypto_data[symbol].empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crypto_data[symbol].index, y=crypto_data[symbol]['close'], mode='lines', name=symbol))
        fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation & Sensitivity Table
# --------------------------------------------------
if len(selected_symbols) > 1 and "BTC-USD" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data, "BTC-USD")
    if not results_df.empty:
        st.markdown("### Correlation & Sensitivity to BTC")
        st.dataframe(results_df, use_container_width=True)



