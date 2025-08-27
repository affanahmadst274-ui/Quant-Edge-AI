import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

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
        trend = "Positive"
    elif slope < 0:
        trend = "Negative"
    else:
        trend = "Flat"

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
# STEP 4 & 5 — Strongest Support & Resistance
# -------------------------------
def calculate_support_resistance(df, window=20):
    if df is None or df.empty:
        return None, None

    df["rolling_min"] = df["low"].rolling(window=window).min()
    df["rolling_max"] = df["high"].rolling(window=window).max()

    strongest_support = df["rolling_min"].min()
    strongest_resistance = df["rolling_max"].max()

    return strongest_support, strongest_resistance


# -------------------------------
# STEP 6 — FINDING BEST EMA
# -------------------------------
def identify_best_ma_ema(df):
    results = []

    for period in range(15, 91, 2):
        ema_col = f'EMA_{period}'
        df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()

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
# STEP 7 — BTC Correlations with Top 50
# -------------------------------
def calculate_correlation_and_sensitivity(base_df, target_df, decimals=4):
    base_pct = base_df['close'].pct_change().dropna()
    target_pct = target_df['close'].pct_change().dropna()
    aligned = pd.concat([base_pct, target_pct], axis=1).dropna()
    aligned.columns = ['Base_pct', 'Target_pct']

    correlation = aligned.corr().iloc[0, 1]

    reg = LinearRegression().fit(aligned[['Base_pct']], aligned['Target_pct'])
    sensitivity = reg.coef_[0]

    idx = np.arange(len(target_df['close'])).reshape(-1,1)
    trend_reg = LinearRegression().fit(idx, target_df['close'].values)
    trend_score = trend_reg.coef_[0]

    correlation_scaled = (correlation + 1) / 2
    sensitivity_scaled = sensitivity / (abs(sensitivity) + 1)
    trend_scaled = (trend_score - min(trend_score, 0)) / (max(trend_score, 0) + abs(min(trend_score, 0)))

    combined_score = 0.31 * correlation_scaled + 0.32 * sensitivity_scaled + 0.37 * trend_scaled

    return (
        round(correlation, decimals),
        round(sensitivity, decimals),
        round(trend_score, decimals),
        round(combined_score, decimals)
    )


# -------------------------------
# STEP 8 — Suggesting Trades
# -------------------------------
def suggest_trades(base_df, target_df, best_metric, trend):
    high = target_df['high'].max()
    low = target_df['low'].min()
    variance = (high - low) / 6
    take_profit = variance
    stop_loss = take_profit / 4

    target_df['signal'] = None
    target_df['take_profit'] = None
    target_df['stop_loss'] = None

    latest_row = target_df.iloc[-1]

    if trend == 'Positive' and latest_row['close'] > latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Long'
        target_df.loc[target_df.index[-1], 'take_profit'] = latest_row[best_metric] + take_profit
        target_df.loc[target_df.index[-1], 'stop_loss'] = latest_row[best_metric] - stop_loss

    elif trend == 'Negative' and latest_row['close'] < latest_row[best_metric]:
        target_df.loc[target_df.index[-1], 'signal'] = 'Short'
        target_df.loc[target_df.index[-1], 'take_profit'] = latest_row[best_metric] - take_profit
        target_df.loc[target_df.index[-1], 'stop_loss'] = latest_row[best_metric] + stop_loss

    return target_df


# -------------------------------
# STEP 9 — Plotting Suggested Trades
# -------------------------------
def plot_candlestick_with_signals(df, metric_list, title, plot_positions=True, future_time_minutes=300, height=500):
    future_time = df['timestamp'].iloc[-1] + timedelta(minutes=future_time_minutes)

    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    )])

    for metric in metric_list:
        if metric in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[metric], mode='lines', name=metric))

    if plot_positions and 'signal' in df.columns:
        last_signal_row = df.iloc[-1]
        if last_signal_row['signal'] == 'Long':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['take_profit'],
                          fillcolor="green", opacity=0.2, line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['stop_loss'],
                          fillcolor="red", opacity=0.2, line_width=0)

            fig.add_trace(go.Scatter(
                x=[last_signal_row['timestamp']],
                y=[last_signal_row['close']],
                mode="markers+text",
                marker=dict(color="green", size=12, symbol="triangle-up"),
                text=["Long Entry"],
                textposition="top center",
                name="Long Signal"
            ))

        elif last_signal_row['signal'] == 'Short':
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['stop_loss'],
                          fillcolor="red", opacity=0.2, line_width=0)
            fig.add_shape(type="rect",
                          x0=last_signal_row['timestamp'], x1=future_time,
                          y0=last_signal_row[metric_list[0]], y1=last_signal_row['take_profit'],
                          fillcolor="green", opacity=0.2, line_width=0)

            fig.add_trace(go.Scatter(
                x=[last_signal_row['timestamp']],
                y=[last_signal_row['close']],
                mode="markers+text",
                marker=dict(color="red", size=12, symbol="triangle-down"),
                text=["Short Entry"],
                textposition="bottom center",
                name="Short Signal"
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price',
        height=height,
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(fixedrange=False)
    )

    return fig


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crypto Data Viewer", layout="wide")
st.sidebar.title("⚡ Crypto Data Options")

# Top 50 list (shortened for clarity, expand if needed)
top_50 = {
    "Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "XRP (XRP)": "XRP-USD",
    "Tether (USDT)": "USDT-USD", "BNB (BNB)": "BNB-USD", "Solana (SOL)": "SOL-USD",
    "USD Coin (USDC)": "USDC-USD", "Dogecoin (DOGE)": "DOGE-USD", "TRON (TRX)": "TRX-USD",
    "Cardano (ADA)": "ADA-USD", "Chainlink (LINK)": "LINK-USD", "Wrapped Bitcoin (WBTC)": "WBTC-USD",
    "Sui (SUI)": "SUI-USD", "Stellar (XLM)": "XLM-USD", "Bitcoin Cash (BCH)": "BCH-USD",
    "Avalanche (AVAX)": "AVAX-USD", "Hedera (HBAR)": "HBAR-USD", "Cronos (CRO)": "CRO-USD",
    "Litecoin (LTC)": "LTC-USD", "Toncoin (TON)": "TON-USD", "Shiba Inu (SHIB)": "SHIB-USD",
    "Uniswap (UNI)": "UNI-USD", "Polkadot (DOT)": "DOT-USD", "Dai (DAI)": "DAI-USD",
    "Aave (AAVE)": "AAVE-USD", "Monero (XMR)": "XMR-USD", "Pepe (PEPE)": "PEPE-USD",
    "Ethena (USDe)": "USDE-USD", "Mantle (MNT)": "MNT-USD", "OKB (OKB)": "OKB-USD",
    "Lido Staked Ether (STETH)": "STETH-USD", "Wrapped staked Ether (WSTETH)": "WSTETH-USD",
    "Hyperliquid (HYPE)": "HYPE-USD", "Wrapped Ether (WESETH)": "WESETH-USD",
    "Sushi (SUSHI)": "SUSHI-USD", "Filecoin (FIL)": "FIL-USD", "Ethereum Classic (ETC)": "ETC-USD",
    "VeChain (VET)": "VET-USD", "Quant (QNT)": "QNT-USD", "Aptos (APT)": "APT-USD",
    "Algorand (ALGO)": "ALGO-USD", "The Graph (GRT)": "GRT-USD", "Immutable X (IMX)": "IMX-USD",
    "Bitget Token (BGB)": "BGB-USD", "Zcash (ZEC)": "ZEC-USD", "Flow (FLOW)": "FLOW-USD",
    "Curve DAO (CRV)": "CRV-USD", "Compound (COMP)": "COMP-USD", "Maker (MKR)": "MKR-USD"
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
show_step_7 = st.sidebar.checkbox("Show Step 7 (BTC Correlations)", value=False)
show_step_8 = st.sidebar.checkbox("Show Step 8 (Suggest Trades)", value=False)

# --- Main Data Fetch ---
df = fetch_crypto_data(symbol, interval, period)

best_coin = None
results_df = None

if df.empty:
    st.error("⚠️ No data available for this selection.")
else:
    current_price = df["close"].iloc[-1]
    trend, y_pred = calculate_regression(df)

    # --- Run Step 7 always if Step 8 is selected ---
    if show_step_7 or show_step_8:
        base_df = fetch_crypto_data("BTC-USD", interval="15m", period="2d")
        results = []
        for coin_name, coin_symbol in top_50.items():
            if coin_symbol == "BTC-USD":
                continue
            target_df = fetch_crypto_data(coin_symbol, interval="15m", period="2d")
            if target_df.empty:
                continue
            corr, sens, trend_scr, combo = calculate_correlation_and_sensitivity(base_df, target_df)
            results.append({
                "Coin": coin_name,
                "Correlation": corr,
                "Sensitivity": sens,
                "Trend Score": trend_scr,
                "Combined Score": combo
            })
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values("Combined Score", ascending=False).reset_index(drop=True)

            btc_trend, _ = calculate_regression(base_df)
            if btc_trend == 'Positive':
                best_coin = results_df.iloc[0]['Coin']
            else:
                best_coin = results_df.sort_values(
                    ['Trend Score', 'Correlation', 'Sensitivity'],
                    ascending=[True, False, False]
                ).iloc[0]['Coin']

    # --- Top row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=f"{symbol_name} Price", value=f"${current_price:,.2f}")
    with col2:
        st.info(f"Trend: **{trend}**" if trend else "Trend: N/A")
    with col3:
        st.metric(label="Best Coin Selected", value=best_coin if best_coin else "N/A")

    # --- Chart ---
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
        best_metric, df = identify_best_ma_ema(df)
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df[best_metric], mode="lines",
            name=f"Best {best_metric}", line=dict(color="magenta", width=2)
        ))

    if show_sr:
        support, resistance = calculate_support_resistance(df, window=20)
        if support:
            fig.add_trace(go.Scatter(
                x=[df["timestamp"].min(), df["timestamp"].max()],
                y=[support, support],
                mode="lines", name=f"Support: {support:.2f}",
                line=dict(color="green", width=2, dash="dot"),
                hovertemplate="Support Price: %{y:.2f}<extra></extra>"
            ))
        if resistance:
            fig.add_trace(go.Scatter(
                x=[df["timestamp"].min(), df["timestamp"].max()],
                y=[resistance, resistance],
                mode="lines", name=f"Resistance: {resistance:.2f}",
                line=dict(color="red", width=2, dash="dot"),
                hovertemplate="Resistance Price: %{y:.2f}<extra></extra>"
            ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Price")
    )

    st.plotly_chart(fig, use_container_width=True)

 # --- Show Step 7 Results (correlations) ---
if show_step_7 and results_df is not None:
    st.subheader("📊 Step 7: BTC Correlations with Top 50 Coins")
    st.dataframe(results_df, use_container_width=True)

# --- STEP 8 & Trade History logic ---
# Compute suggested trade for best_coin (if requested)
latest_row = None
best_coin_df = None

if show_step_8 and best_coin:
    st.subheader("📈 Step 8: Suggested Trades")
    best_symbol = top_50[best_coin]
    best_coin_df = fetch_crypto_data(best_symbol, '15m', '2d')

    if best_coin_df.empty:
        st.warning(f"No data for {best_coin}")
    else:
        best_coin_df['EMA_20'] = best_coin_df['close'].ewm(span=20, adjust=False).mean()
        best_coin_df = suggest_trades(base_df, best_coin_df, 'EMA_20', btc_trend)
        latest_row = best_coin_df.iloc[-1]

        # Build and show latest-signal table (Entry Price as N/A when no signal)
        sig = latest_row.get('signal', None)
        latest_signal_display = pd.DataFrame([{
            "Entry Price": f"${latest_row['close']:.6f}" if pd.notnull(sig) and sig else "N/A",
            "Signal": sig if pd.notnull(sig) and sig else "N/A",
            "Take Profit": f"${latest_row['take_profit']:.6f}" if pd.notnull(latest_row.get('take_profit')) else "N/A",
            "Stop Loss": f"${latest_row['stop_loss']:.6f}" if pd.notnull(latest_row.get('stop_loss')) else "N/A"
        }])
        st.dataframe(latest_signal_display, use_container_width=True)

        # If a new signal exists, append trade to session_state history (avoid duplicates)
        if sig and sig != "N/A":
            entry_price = float(latest_row['close'])
            tp = float(latest_row['take_profit']) if pd.notnull(latest_row.get('take_profit')) else np.nan
            sl = float(latest_row['stop_loss']) if pd.notnull(latest_row.get('stop_loss')) else np.nan

            exists = False
            if not st.session_state.trade_history.empty:
                # compare coin, signal and entry price approximately
                exists = (
                    (st.session_state.trade_history["Coin"] == best_coin) &
                    (st.session_state.trade_history["Signal"] == sig) &
                    (np.isclose(st.session_state.trade_history["Entry Price"].astype(float), entry_price, atol=1e-8))
                ).any()

            if not exists:
                new_trade = {
                    "Coin": best_coin,
                    "Entry Price": entry_price,
                    "Signal": sig,
                    "Take Profit": tp,
                    "Stop Loss": sl,
                    "Current Price": entry_price,
                    "PnL %": 0.0,
                    "Status": "Running",
                    "Timestamp": latest_row["timestamp"]
                }
                st.session_state.trade_history = pd.concat(
                    [st.session_state.trade_history, pd.DataFrame([new_trade])],
                    ignore_index=True
                )

# Update running trades' Current Price, PnL %, and Status (TP/SL detection)
if not st.session_state.trade_history.empty:
    for idx in st.session_state.trade_history.index:
        trade = st.session_state.trade_history.loc[idx]
        coin_symbol = top_50.get(trade["Coin"])
        if coin_symbol is None:
            continue
        coin_df = fetch_crypto_data(coin_symbol, "15m", "2d")
        if coin_df.empty:
            continue
        current_price = float(coin_df["close"].iloc[-1])
        st.session_state.trade_history.at[idx, "Current Price"] = current_price

        entry = float(trade["Entry Price"])
        tp = float(trade["Take Profit"]) if pd.notnull(trade["Take Profit"]) else np.nan
        sl = float(trade["Stop Loss"]) if pd.notnull(trade["Stop Loss"]) else np.nan

        if trade["Signal"] == "Long":
            pnl = (current_price - entry) / entry * 100 if entry != 0 else np.nan
            status = "Running"
            if pd.notnull(tp) and current_price >= tp:
                status = "TP Hit"
            elif pd.notnull(sl) and current_price <= sl:
                status = "SL Hit"
        else:  # Short
            pnl = (entry - current_price) / entry * 100 if entry != 0 else np.nan
            status = "Running"
            if pd.notnull(tp) and current_price <= tp:
                status = "TP Hit"
            elif pd.notnull(sl) and current_price >= sl:
                status = "SL Hit"

        st.session_state.trade_history.at[idx, "PnL %"] = round(pnl, 2) if pd.notnull(pnl) else np.nan
        st.session_state.trade_history.at[idx, "Status"] = status

# Display trade history (formatted) under Step 7 if requested
if show_step_7:
    st.subheader("📊 Step 7: Trade History & Performance")
    if not st.session_state.trade_history.empty:
        display_hist = st.session_state.trade_history.copy()
        # formatting for readability (adjust decimals as you like)
        display_hist["Entry Price"] = display_hist["Entry Price"].apply(lambda x: f"${x:,.6f}" if pd.notnull(x) else "N/A")
        display_hist["Current Price"] = display_hist["Current Price"].apply(lambda x: f"${x:,.6f}" if pd.notnull(x) else "N/A")
        display_hist["Take Profit"] = display_hist["Take Profit"].apply(lambda x: f"${x:,.6f}" if pd.notnull(x) else "N/A")
        display_hist["Stop Loss"] = display_hist["Stop Loss"].apply(lambda x: f"${x:,.6f}" if pd.notnull(x) else "N/A")
        display_hist["PnL %"] = display_hist["PnL %"].apply(lambda x: f\"{x:.2f}%\" if pd.notnull(x) else "N/A")
        st.dataframe(display_hist, use_container_width=True)
    else:
        st.info("No trades recorded yet.")

# --- Step 9: Suggested Trade Chart (only once, outside loop) ---
if show_step_8 and best_coin and (latest_row is not None) and pd.notnull(latest_row.get("signal")) and latest_row.get("signal"):
    st.subheader("📉 Step 9: Suggested Trade Chart")
    trade_fig = plot_candlestick_with_signals(
        best_coin_df,
        ['EMA_20'],
        f"{best_coin} with Suggested Position",
        future_time_minutes=300,
        height=500
    )
    st.plotly_chart(trade_fig, use_container_width=True)

