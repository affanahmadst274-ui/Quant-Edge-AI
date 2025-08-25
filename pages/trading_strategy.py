import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------
# Fetch Crypto Data (Yahoo Finance Compatible)
# -------------------------------------------
def fetch_crypto_data(symbol, interval="1d", days=180):
    try:
        df = yf.download(symbol, period=f"{days}d", interval=interval)

        if df.empty:
            return pd.DataFrame()

        # Reset index (Date becomes column)
        df = df.reset_index()

        # Standardize column names
        df.rename(
            columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # Keep only required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# -------------------------------------------
# Prediction Model (Simple Linear Regression)
# -------------------------------------------
def train_and_predict(df):
    try:
        df["returns"] = df["close"].pct_change(fill_method=None)
        df = df.dropna()

        X = np.arange(len(df)).reshape(-1, 1)
        y = df["close"].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        future_X = np.arange(len(df), len(df) + 7).reshape(-1, 1)
        predictions = model.predict(future_X)

        future_dates = pd.date_range(start=df["timestamp"].iloc[-1], periods=7, freq="D")

        pred_df = pd.DataFrame({"timestamp": future_dates, "predicted_close": predictions})
        return pred_df

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return pd.DataFrame()

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
def main():
    st.set_page_config(page_title="Crypto Price Prediction", layout="wide")

    st.title("📈 Crypto Price Prediction App")
    st.write("Get crypto prices and simple predictions using Yahoo Finance data.")

    # Sidebar Inputs
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Enter Symbol (e.g., BTC-USD, ETH-USD)", value="BTC-USD")
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m"])
    days = st.sidebar.slider("Days of Data", min_value=30, max_value=365, value=180, step=30)

    # Auto-fetch data when settings change
    df = fetch_crypto_data(symbol, interval, days)

    if df.empty:
        st.error("⚠️ No data fetched. Try a different symbol or interval.")
        return

    st.subheader(f"📊 Historical Data for {symbol}")
    st.dataframe(df.tail(10))

    # Plot Closing Prices
    st.line_chart(df.set_index("timestamp")["close"])

    # Train model and predict
    st.subheader("🔮 Predicted Prices (Next 7 Days)")
    pred_df = train_and_predict(df)

    if not pred_df.empty:
        st.dataframe(pred_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["timestamp"], df["close"], label="Historical")
        ax.plot(pred_df["timestamp"], pred_df["predicted_close"], label="Predicted", linestyle="--")
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()


















