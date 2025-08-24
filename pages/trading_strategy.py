# pages/trading_strategy.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Strategy", layout="wide")
st.title("📈 Trading Strategy Analysis")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Strategy Settings")
ticker = st.sidebar.text_input("Enter Crypto Symbol", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
ema_periods = st.sidebar.multiselect("EMA Periods", [10, 20, 50, 100, 200], default=[20, 50, 100])

# -----------------------------
# Download & normalize data
# -----------------------------
@st.cache_data(ttl=300, show_spinner=True)
def load_price_data(tkr: str, start, end) -> pd.DataFrame:
    # Force single-level columns from yfinance
    df = yf.download(
        tkr,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",  # critical to avoid MultiIndex for single tickers
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # If MultiIndex still sneaks in, reduce to single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        # Try common layouts and reduce to the requested ticker
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if "Close" in lvl0 and tkr in lvl1:
            df = df.xs(tkr, axis=1, level=1)
        elif tkr in lvl0 and "Close" in lvl1:
            df = df.xs(tkr, axis=1, level=0)
        else:
            # Fallback: just drop a level (works when only one ticker is present)
            try:
                df = df.droplevel(0, axis=1)
            except Exception:
                df = df.droplevel(1, axis=1)

    # Make sure the essential columns exist
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        # Some intervals may not include all columns; try to reconstruct Close at least
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
            missing -= {"Close"}
        # If anything still missing, return empty to handle gracefully
        if missing:
            return pd.DataFrame()

    # Ensure 'Close' is a Series (not a single-column DataFrame)
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]

    return df

df = load_price_data(ticker, start_date, end_date)

if df.empty:
    st.error("❌ No data found for the given ticker and date range.")
    st.stop()

# -----------------------------
# Compute EMAs and signals
# -----------------------------
for period in ema_periods:
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

# Use a primary EMA for signals (first from selection)
primary_period = ema_periods[0]
primary_col = f"EMA_{primary_period}"

# Compare Series-to-Series directly (no align needed)
df["Signal"] = 0
df.loc[df["Close"] > df[primary_col], "Signal"] = 1
df.loc[df["Close"] <= df[primary_col], "Signal"] = -1

# -----------------------------
# Plot: Price + EMAs
# -----------------------------
fig_price, ax_price = plt.subplots(figsize=(12, 6))
ax_price.plot(df.index, df["Close"], label="Close", linewidth=1.3)
for period in ema_periods:
    ax_price.plot(df.index, df[f"EMA_{period}"], linestyle="--", label=f"EMA {period}", linewidth=1.0)
ax_price.set_title(f"{ticker} — Price with EMAs (Signal uses EMA {primary_period})")
ax_price.legend()
st.pyplot(fig_price)

# -----------------------------
# Backtest: simple EMA crossover (long when Close > EMA)
# -----------------------------
df["Market Return"] = df["Close"].pct_change(fill_method=None)
df["Strategy Return"] = df["Signal"].shift(1) * df["Market Return"]

cum_mkt = (1 + df["Market Return"].fillna(0)).cumprod()
cum_strat = (1 + df["Strategy Return"].fillna(0)).cumprod()

fig_ret, ax_ret = plt.subplots(figsize=(12, 6))
ax_ret.plot(cum_mkt, label="Market Return")
ax_ret.plot(cum_strat, label="Strategy Return")
ax_ret.set_title(f"{ticker} — Cumulative Returns")
ax_ret.legend()
st.pyplot(fig_ret)

# -----------------------------
# Metrics & Insights
# -----------------------------
st.subheader("📊 Strategy Metrics")

final_mkt = float(cum_mkt.iloc[-1])
final_strat = float(cum_strat.iloc[-1])
outperf_pct = (final_strat - final_mkt) * 100.0

m1, m2, m3 = st.columns(3)
m1.metric("Final Market Value", f"{final_mkt:.2f}×")
m2.metric("Final Strategy Value", f"{final_strat:.2f}×")
m3.metric("Outperformance", f"{outperf_pct:.2f}%")

# % of time price is above each EMA
st.markdown("### EMA Signals")
for period in ema_periods:
    pct_above = (df["Close"] > df[f"EMA_{period}"]).mean() * 100.0
    st.metric(label=f"Above EMA {period}", value=f"{pct_above:.2f}%")

st.success("✅ Trading Strategy page loaded successfully.")


