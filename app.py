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
cryptos = st.sidebar.multiselect(
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

data = {}
for crypto in cryptos:
    data[crypto] = yf.download(crypto, start=start, end=end)

# ----------------------------
# Banner Section (Latest Prices)
# ----------------------------
st.markdown("## 📌 Latest Prices")
cols = st.columns(len(cryptos))
for i, crypto in enumerate(cryptos):
    latest_price = round(data[crypto]["Close"].iloc[-1], 2)
    cols[i].metric(crypto.replace("-USD", ""), f"${latest_price}")

# ----------------------------
# Candlestick Chart
# ----------------------------
st.markdown("## 📈 Candlestick Chart")
for crypto in cryptos:
    st.subheader(crypto.replace("-USD", ""))
    df = data[crypto].copy()
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

# ----------------------------
# Correlation & Sensitivity
# ----------------------------
st.markdown("## 🔗 Correlation & Sensitivity Analysis")

# Combine closing prices
closes = pd.DataFrame({crypto: data[crypto]["Close"] for crypto in cryptos})

# --- Correlation Table ---
corr = closes.corr()
st.subheader("Correlation Table")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.dataframe(corr.style.background_gradient(cmap="RdBu", axis=None), use_container_width=True)
except ImportError:
    st.dataframe(corr, use_container_width=True)

# --- Correlation Heatmap ---
try:
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="RdBu", center=0, ax=ax)
    st.pyplot(fig)
except Exception:
    st.info("Correlation heatmap not available (matplotlib/seaborn missing).")

# --- Sensitivity Table (vs BTC) ---
if "BTC-USD" in closes.columns:
    st.subheader("Sensitivity Table (vs BTC)")
    sensitivity = closes.pct_change().corrwith(closes["BTC-USD"])
    sensitivity_df = pd.DataFrame(sensitivity, columns=["Sensitivity vs BTC"])
    st.dataframe(sensitivity_df, use_container_width=True)
else:
    st.warning("BTC-USD is not in selection, sensitivity table skipped.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly & yFinance")







