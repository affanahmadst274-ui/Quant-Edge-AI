import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_data
def download_crypto_data(symbols, period="6mo", interval="1d"):
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, period=period, interval=interval)
            if not df.empty:
                df.reset_index(inplace=True)
                df.rename(
                    columns={
                        "Date": "timestamp",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
                data[symbol] = df
        except Exception as e:
            st.warning(f"⚠️ Could not fetch {symbol}: {e}")
    return data


def predict_pair_value(base_df, target_df):
    try:
        merged = pd.merge(base_df[["timestamp", "close"]],
                          target_df[["timestamp", "close"]],
                          on="timestamp",
                          suffixes=("_base", "_target"))
        X = merged[["close_base"]]
        y = merged["close_target"]

        model = LinearRegression()
        model.fit(X, y)

        latest_base = base_df["close"].iloc[-1]
        prediction = model.predict([[latest_base]])[0]
        return prediction
    except Exception:
        return None


def calculate_correlation_and_sensitivity_relative_to_base(data, base_symbol):
    try:
        if base_symbol not in data or data[base_symbol].empty:
            return pd.DataFrame()

        base = data[base_symbol][["timestamp", "close"]].rename(
            columns={"close": "base_close"}
        )
        results = []
        for symbol, df in data.items():
            if symbol == base_symbol or df.empty:
                continue
            merged = pd.merge(base, df[["timestamp", "close"]],
                              on="timestamp", suffixes=("", "_other"))
            if merged.empty:
                continue
            correlation = merged["base_close"].corr(merged["close"])
            sensitivity = (
                merged["close"].pct_change().corr(merged["base_close"].pct_change())
            )
            results.append({
                "Symbol": symbol,
                "Correlation to BTC": correlation,
                "Sensitivity to BTC": sensitivity
            })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

st.image("Pic2.PNG", use_container_width=True)
st.sidebar.image("Pic1.PNG", use_container_width=True)

st.sidebar.header("Select Cryptocurrencies")
all_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
selected_symbols = st.sidebar.multiselect(
    "Choose cryptos", all_symbols, default=["BTC-USD", "ETH-USD"]
)

target_symbol = st.sidebar.selectbox(
    "Target crypto for prediction", selected_symbols
)

crypto_data = download_crypto_data([s.replace("-", "") for s in selected_symbols])

# --------------------------------------------------
# KPIs
# --------------------------------------------------
st.markdown("## Crypto Strength Analysis, Correlations and Predictions")
kpi1, kpi2, kpi3 = st.columns(3)

btc_price = (
    crypto_data["BTCUSD"]["close"].iloc[-1].item()
    if "BTCUSD" in crypto_data and not crypto_data["BTCUSD"].empty
    else 0.0
)

target_price_val = (
    crypto_data[target_symbol.replace("-", "")]["close"].iloc[-1].item()
    if target_symbol.replace("-", "") in crypto_data
    and not crypto_data[target_symbol.replace("-", "")].empty
    else 0.0
)

predicted_price = None
if (
    "BTCUSD" in crypto_data
    and target_symbol.replace("-", "") in crypto_data
    and not crypto_data["BTCUSD"].empty
    and not crypto_data[target_symbol.replace("-", "")].empty
):
    predicted_price = predict_pair_value(
        crypto_data["BTCUSD"], crypto_data[target_symbol.replace("-", "")]
    )

kpi1.metric("BTC Price", f"${btc_price:,.2f}")
kpi2.metric(f"{target_symbol} Price", f"${target_price_val:,.2f}")
if predicted_price:
    kpi3.metric("Predicted Price", f"${predicted_price:,.2f}")
else:
    kpi3.metric("Predicted Price", "N/A")

# --------------------------------------------------
# Candlestick Charts
# --------------------------------------------------
st.markdown("### Candlestick Charts")
for symbol in selected_symbols:
    clean_symbol = symbol.replace("-", "")
    if clean_symbol in crypto_data and not crypto_data[clean_symbol].empty:
        df = crypto_data[clean_symbol]
        fig = go.Figure(data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"]
            )
        ])
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title=None,
            yaxis_title=None,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No data for {symbol}")

# --------------------------------------------------
# Correlation & Sensitivity Table
# --------------------------------------------------
if len(selected_symbols) > 1 and "BTC-USD" in selected_symbols:
    results_df = calculate_correlation_and_sensitivity_relative_to_base(
        crypto_data, "BTCUSD"
    )
    if not results_df.empty:
        st.markdown("### Correlation & Sensitivity to BTC")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No correlation data available.")




