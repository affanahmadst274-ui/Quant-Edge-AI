import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📊 Trading Strategy Analysis")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Strategy Parameters")

ticker = st.sidebar.text_input("Enter Ticker (Yahoo format)", "BTC-USD")

start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

ema_periods = st.sidebar.multiselect(
    "Select EMA Periods", [20, 50, 100, 200], default=[20, 50, 200]
)

# -----------------------------
# Load Data
# -----------------------------
df = yf.download(ticker, start=start_date, end=end_date, interval="1d")

if df.empty:
    st.error("No data found for the selected ticker and date range.")
    st.stop()

df["Return"] = df["Adj Close"].pct_change(fill_method=None)

# -----------------------------
# EMA Strategy
# -----------------------------
ema_scores = {}
for period in ema_periods:
    df[f"EMA_{period}"] = df["Adj Close"].ewm(span=period, adjust=False).mean()
    score = (df["Adj Close"] > df[f"EMA_{period}"]).sum() / len(df) * 100
    ema_scores[period] = score

# -----------------------------
# Linear Regression Trend
# -----------------------------
df = df.dropna()
X = np.arange(len(df)).reshape(-1, 1)
y = df["Adj Close"].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

df["Trend"] = trend

# -----------------------------
# Volatility Measure
# -----------------------------
volatility = df["Return"].std() * np.sqrt(252)  # annualized volatility

# -----------------------------
# Display Results
# -----------------------------
st.subheader(f"Trading Strategy for {ticker}")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### EMA Signals")
    for period, score in ema_scores.items():
        # Ensure scalar value for formatting
        if isinstance(score, pd.Series):
            score = float(score.iloc[0])
        else:
            score = float(score)
        st.metric(label=f"Above EMA {period}", value=f"{score:.2f}%")

with col2:
    st.markdown("### Trend")
    slope = model.coef_[0][0]
    trend_label = "📈 Uptrend" if slope > 0 else "📉 Downtrend"
    st.metric(label="Trend Direction", value=f"{trend_label}")

with col3:
    st.markdown("### Volatility")
    st.metric(label="Annualized Volatility", value=f"{volatility:.2%}")

# -----------------------------
# Charts
# -----------------------------
st.markdown("### Price vs Trend and EMAs")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Adj Close"], label="Price", color="blue")
ax.plot(df.index, df["Trend"], label="Trend", color="red", linestyle="--")

for period in ema_periods:
    ax.plot(df.index, df[f"EMA_{period}"], label=f"EMA {period}")

ax.legend()
st.pyplot(fig)

# -----------------------------
# Correlation with Market
# -----------------------------
st.markdown("### Correlation with Bitcoin Market")

if ticker != "BTC-USD":
    market_df = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d")
    market_df["Return"] = market_df["Adj Close"].pct_change(fill_method=None)
    combined = pd.concat([df["Return"], market_df["Return"]], axis=1)
    combined.columns = [ticker, "BTC"]
    correlation = combined.corr().iloc[0, 1]
    st.metric(label="Correlation with BTC", value=f"{correlation:.2f}")
else:
    st.info("Correlation skipped because the selected ticker is BTC itself.")



