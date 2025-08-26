import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------
# Function to fetch crypto data
# -------------------------------
def fetch_crypto_data(symbol: str, interval: str, period: str):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.rename(
            columns={
                "Datetime": "timestamp",
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            },
            inplace=True,
        )
        df = df[["timestamp", "open", "high", "low", "close"]]

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
        return pd.DataFrame()


# -------------------------------
# STEP 2 — TREND REGRESSION
# -------------------------------
def calculate_regression(df):
    if df is None or df.empty or "close" not in df.columns:
        return None, None

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    slope = model.coef_[0]

    if slope > 0:
        trend = "📈 Positive Trend"
    elif slope < 0:
        trend = "📉 Negative Trend"
    else:
        trend = "➡️ Flat Trend"

    return trend, y_pred


# -------------------------------
# STEP 3 — REGRESSION PLOT
# -------------------------------
def plot_regression(df, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["close"], mode="lines",
        name="Close Price", line=dict(color="blue")
    ))

    if y_pred is not None:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=y_pred, mode="lines",
            name="Regression Line", line=dict(color="red", dash="dot")
        ))

    fig.update_layout(
        title="📊 Close Price with Regression Line",
        xaxis_title="Time", yaxis_title="Price",
        template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


# -------------------------------
# STEP 4 & 5 — Support & Resistance
# -------------------------------
def calculate_support_resistance(df, window=20):
    """Find support and resistance using rolling min/max."""
    if df is None or df.empty:
        return [], []

    df["rolling_min"] = df["low"].rolling(window=window).min()
    df["rolling_max"] = df["high"].rolling(window=window).max()

    # Support = local minima, Resistance = local maxima
    support_levels = df["rolling_min"].dropna().unique().tolist()
    resistance_levels = df["rolling_max"].dropna().unique().tolist()

    return support_levels, resistance_levels


# -------------------------------
# STEP 6 — FINDING BEST EMA
# -------------------------------
def identify_best_ma_ema(df):
    results = []

    for period in range(15, 91, 2):
        ema_col = f'EMA_{period}'
        df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()

        # Placeholder scoring
        velocity = np.random.random()
        efficiency = np.random.random()

        results.append({
            'Period': period,
            'EMA': ema_col,
            'Velocity': velocity,
            'Efficiency': efficiency
        })

    results_df = pd.DataFrame(results)
    results_df['Combined Score'] = (
        0.41 * results_df['Velocity'] + 0.59 * results_df['Efficiency']
    )
    best = results_df.sort_values('Combined Score', ascending=False).iloc[0]
    return best['EMA'], df


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crypto Data Viewer", layout="wide")
st.sidebar.title("⚡ Crypto Data Options")

top_50 = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "Binance Coin (BNB)": "BNB-USD",
    "Solana (SOL)": "SOL-USD", "XRP": "XRP-USD", "Cardano (ADA)": "ADA-USD",
    "Dogecoin (DOGE)": "DOGE-USD", "Polkadot (DOT)": "DOT-USD", "Polygon (MATIC)": "MATIC-USD",
    "Litecoin (LTC)": "LTC-USD", "Shiba Inu (SHIB)": "SHIB-USD", "TRON (TRX)": "TRX-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Uniswap (UNI)": "UNI-USD", "Chainlink (LINK)": "LINK-USD",
    "Cosmos (ATOM)": "ATOM-USD", "Monero (XMR)": "XMR-USD", "Stellar (XLM)": "XLM-USD",
    "OKB": "OKB-USD", "Toncoin (TON)": "TON-USD", "Ethereum Classic (ETC)": "ETC-USD",
    "Bitcoin Cash (BCH)": "BCH-USD", "Filecoin (FIL)": "FIL-USD", "Lido DAO (LDO)": "LDO-USD",
    "Aptos (APT)": "APT-USD", "Hedera (HBAR)": "HBAR-USD", "Cronos (CRO)": "CRO-USD",
    "Arbitrum (ARB)": "ARB-USD", "VeChain (VET)": "VET-USD", "NEAR Protocol (NEAR)": "NEAR-USD",
    "Optimism (OP)": "OP-USD", "Maker (MKR)": "MKR-USD", "Algorand (ALGO)": "ALGO-USD",
    "Synthetix (SNX)": "SNX-USD", "Render (RNDR)": "RNDR-USD", "Quant (QNT)": "QNT-USD",
    "Aave (AAVE)": "AAVE-USD", "The Graph (GRT)": "GRT-USD", "Stacks (STX)": "STX-USD",
    "Immutable (IMX)": "IMX-USD", "Fantom (FTM)": "FTM-USD", "Tezos (XTZ)": "XTZ-USD",
    "Theta Network (THETA)": "THETA-USD", "Axie Infinity (AXS)": "AXS-USD", "EOS": "EOS-USD",
    "Flow": "FLOW-USD", "Kava (KAVA)": "KAVA-USD", "Chiliz (CHZ)": "CHZ-USD"
}

symbol_name = st.sidebar.selectbox("Select Coin", list(top_50.keys()))
symbol = top_50[symbol_name]

interval = st.sidebar.selectbox("Select Interval",
    ["5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=3)

period = st.sidebar.selectbox("Select Period",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=2)

chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line", "Regression"], index=0)

show_step_6 = st.sidebar.checkbox("Show Step 6 (Best EMA Finder)", value=False)
show_sr = st.sidebar.checkbox("Show Support & Resistance", value=False)

df = fetch_crypto_data(symbol, interval, period)

if df.empty:
    st.error("⚠️ No data available for this selection.")
else:
    current_price = df["close"].iloc[-1]
    trend, y_pred = calculate_regression(df)

    # --- Top row with Price, Trend, Best EMA ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"{symbol_name} Price", value=f"${current_price:,.2f}")
    with col2:
        st.info(f"Trend: **{trend}**" if trend else "Trend: N/A")
    with col3:
        if show_step_6:
            best_metric, df = identify_best_ma_ema(df)
            st.success(f"🏆 Best EMA: **{best_metric}**")

    # --- Chart rendering ---
    if chart_type == "Candlestick":
        fig = go.Figure([go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price"
        )])
    elif chart_type == "Line":
        fig = go.Figure([go.Scatter(
            x=df["timestamp"], y=df["close"], mode="lines", name="Close Price"
        )])
    else:
        fig = plot_regression(df, y_pred)

    if chart_type in ["Candlestick", "Line"] and y_pred is not None:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=y_pred, mode="lines",
            name="Trend Line", line=dict(color="black", width=2, dash="dot")
        ))

    if show_step_6:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[best_metric], mode="lines",
            name=f"Best {best_metric}", line=dict(color="magenta", width=2)
        ))

    if show_sr:
        supports, resistances = calculate_support_resistance(df, window=20)
        for s in supports[-3:]:  # show last 3 support levels
            fig.add_hline(y=s, line=dict(color="green", width=1, dash="dot"), annotation_text="Support")
        for r in resistances[-3:]:  # show last 3 resistance levels
            fig.add_hline(y=r, line=dict(color="red", width=1, dash="dot"), annotation_text="Resistance")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Price")
    )

    st.plotly_chart(fig, use_container_width=True)



















