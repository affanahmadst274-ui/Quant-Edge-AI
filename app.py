import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

# ----------------------------
# Title
# ----------------------------
st.title("📊 Crypto Price Dashboard")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Settings")
selected_symbols = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    default=["BTC-USD", "ETH-USD"]
)

period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# ----------------------------
# Fetch Data
# ----------------------------
end = datetime.today()
start = end - timedelta(days=365*2)

crypto_data = {}
for symbol in selected_symbols:
    crypto_data[symbol] = yf.download(symbol, start=start, end=end)

# ----------------------------
# Banner Section (Latest Prices)
# ----------------------------
st.markdown("## 📌 Latest Prices")
cols = st.columns(len(selected_symbols))
for i, symbol in enumerate(selected_symbols):
    latest_price = round(crypto_data[symbol]["Close"].iloc[-1], 2)
    cols[i].metric(symbol.replace("-USD", ""), f"${latest_price}")

# ----------------------------
# Candlestick Chart
# ----------------------------
st.markdown("## 📈 Candlestick Chart")
for symbol in selected_symbols:
    st.subheader(symbol.replace("-USD", ""))
    df = crypto_data[symbol].copy()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    fig.update_layout(height=400, width=900, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Correlation & Sensitivity Table
# --------------------------------------------------
def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    closes = pd.DataFrame({sym: data[sym]["Close"] for sym in data})
    returns = closes.pct_change().dropna()

    corr = returns.corr()[base_symbol]
    sens = returns.corrwith(returns[base_symbol])

    result = pd.DataFrame({
        "Correlation to BTC": corr,
        "Sensitivity to BTC": sens
    })
    return result

if len(selected_symbols) > 1 and "BTC-USD" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(crypto_data, "BTC-USD")
    if not results_df.empty:
        st.markdown("### Correlation & Sensitivity to BTC")
        st.dataframe(results_df, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly & yFinance")


