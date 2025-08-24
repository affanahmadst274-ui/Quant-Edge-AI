import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Trading Strategy", layout="wide")

st.title("📈 Trading Strategy")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter a Crypto Ticker (e.g., BTC-USD)", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty:
    st.warning("No data found. Please check the ticker symbol.")
    st.stop()

# Technical Indicators
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()

# Fix RSI calculation (avoid Series error + deprecation warning)
returns = df['Close'].pct_change(fill_method=None)
df['RSI'] = 100 - (100 / (1 + returns.rolling(14).mean()))

# Strategy: EMA Crossover
def generate_signals(data):
    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['EMA20'].iloc[i] > data['EMA50'].iloc[i] and not position:
            buy_signals.append(data['Close'].iloc[i])
            sell_signals.append(np.nan)
            position = True
        elif data['EMA20'].iloc[i] < data['EMA50'].iloc[i] and position:
            buy_signals.append(np.nan)
            sell_signals.append(data['Close'].iloc[i])
            position = False
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    data['Buy'] = buy_signals
    data['Sell'] = sell_signals
    return data

df = generate_signals(df)

# Linear Regression Prediction
model = LinearRegression()
df['Days'] = np.arange(len(df))
X = df[['Days']]
y = df['Close']
model.fit(X, y)

future_days = 30
future_index = np.array(range(len(df), len(df) + future_days)).reshape(-1, 1)
future_preds = model.predict(future_index)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50'))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA100'], name='EMA100'))
fig.add_trace(go.Scatter(x=df.index, y=df['Buy'], mode='markers', name='Buy Signal',
                         marker=dict(color='green', size=10, symbol='triangle-up')))
fig.add_trace(go.Scatter(x=df.index, y=df['Sell'], mode='markers', name='Sell Signal',
                         marker=dict(color='red', size=10, symbol='triangle-down')))

# Future Predictions Line
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Predicted Price",
                         line=dict(dash='dot', color='orange')))

fig.update_layout(title=f"{ticker} Trading Strategy",
                  xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# KPIs Section
st.subheader("📊 Key Performance Insights")
col1, col2 = st.columns(2)

# EMA Performance (FIX: convert to float)
ema_periods = [20, 50, 100]
ema_scores = {}
for period in ema_periods:
    ema = df['Close'].ewm(span=period, adjust=False).mean()
    above = (df['Close'] > ema).mean() * 100  # this is a scalar float
    ema_scores[period] = float(above)  # ensure scalar

with col1:
    st.markdown("### EMA Signals")
    for period, score in ema_scores.items():
        st.metric(label=f"Above EMA {period}", value=f"{score:.2f}%")

# Correlation with Market
with col2:
    st.markdown("### Correlation with Market")
    try:
        market = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
        corr = df['Close'].pct_change(fill_method=None).corr(market['Close'].pct_change(fill_method=None))
        st.metric(label="Correlation with BTC-USD", value=f"{float(corr):.2f}")
    except Exception as e:
        st.warning(f"Could not fetch market data: {e}")


