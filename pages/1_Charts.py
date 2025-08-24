import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.title("🔗 Crypto Correlation Heatmap")

symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
data = {sym: yf.Ticker(sym).history(period="90d", interval="1d")["Close"] for sym in symbols}

df = pd.DataFrame(data)
corr = df.corr()

fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)
