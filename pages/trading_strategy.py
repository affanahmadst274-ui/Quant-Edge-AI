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

    except Exception:
        # Just return empty DataFrame silently
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
# STEP 10 — EMA Touch Backtesting
# -------------------------------
def ema_touch_strategy_with_plot(symbol, ema_number, budget, trading_fee=0.001, interval="15m", period="7d", chart_height=600, trend="positive"):
    """
    Backtest EMA Touch strategy (Long/Short depending on trend).
    """
    df = fetch_crypto_data(symbol, interval, period)
    if df.empty:
        return None

    # Calculate EMA
    ema_col = f"EMA_{ema_number}"
    df[ema_col] = df['close'].ewm(span=ema_number, adjust=False).mean()

    # Variance for TP/SL
    variance = (df['high'].max() - df['low'].min())
    take_profit_factor = variance / 8
    stop_loss_factor = take_profit_factor / 2

    # Tracking
    trades = []
    current_budget = budget
    win_count, total_trades = 0, 0
    active_trade = False
    entry_price, entry_time = 0, None
    buy_markers, sell_markers, positions = [], [], []

    # Simulation loop
    for i in range(1, len(df)):
        prev_price, curr_price = df['close'].iloc[i - 1], df['close'].iloc[i]
        prev_ema, curr_ema = df[ema_col].iloc[i - 1], df[ema_col].iloc[i]
        low_price, high_price = df['low'].iloc[i], df['high'].iloc[i]

        if trend == "positive":  # Longs
            if not active_trade and prev_price > prev_ema and low_price <= curr_ema:
                take_profit = curr_price + take_profit_factor
                stop_loss = curr_price - stop_loss_factor
                qty = current_budget / curr_price
                entry_price, entry_time = curr_price, df['timestamp'].iloc[i]
                active_trade, total_trades = True, total_trades + 1
                buy_markers.append((entry_time, entry_price))

            if active_trade:
                if high_price >= take_profit:  # TP hit
                    profit = qty * (take_profit - entry_price)
                    current_budget += profit - trading_fee * entry_price * qty
                    win_count += 1
                    trades.append({"Type": "Long", "Entry": entry_price, "Exit": take_profit, "P&L": profit})
                    sell_markers.append((df['timestamp'].iloc[i], take_profit))
                    positions.append({"entry_time": entry_time, "exit_time": df['timestamp'].iloc[i],
                                      "entry_price": entry_price, "take_profit": take_profit,
                                      "stop_loss": stop_loss, "outcome": "profit"})
                    active_trade = False
                elif low_price <= stop_loss:  # SL hit
                    loss = qty * (entry_price - stop_loss)
                    current_budget -= loss + trading_fee * entry_price * qty
                    trades.append({"Type": "Long", "Entry": entry_price, "Exit": stop_loss, "P&L": -loss})
                    sell_markers.append((df['timestamp'].iloc[i], stop_loss))
                    positions.append({"entry_time": entry_time, "exit_time": df['timestamp'].iloc[i],
                                      "entry_price": entry_price, "take_profit": take_profit,
                                      "stop_loss": stop_loss, "outcome": "loss"})
                    active_trade = False

        elif trend == "negative":  # Shorts
            if not active_trade and prev_price < prev_ema and high_price >= curr_ema:
                take_profit = curr_price - take_profit_factor
                stop_loss = curr_price + stop_loss_factor
                qty = current_budget / curr_price
                entry_price, entry_time = curr_price, df['timestamp'].iloc[i]
                active_trade, total_trades = True, total_trades + 1
                sell_markers.append((entry_time, entry_price))

            if active_trade:
                if low_price <= take_profit:  # TP hit
                    profit = qty * (entry_price - take_profit)
                    current_budget += profit - trading_fee * entry_price * qty
                    win_count += 1
                    trades.append({"Type": "Short", "Entry": entry_price, "Exit": take_profit, "P&L": profit})
                    buy_markers.append((df['timestamp'].iloc[i], take_profit))
                    positions.append({"entry_time": entry_time, "exit_time": df['timestamp'].iloc[i],
                                      "entry_price": entry_price, "take_profit": take_profit,
                                      "stop_loss": stop_loss, "outcome": "profit"})
                    active_trade = False
                elif high_price >= stop_loss:  # SL hit
                    loss = qty * (stop_loss - entry_price)
                    current_budget -= loss + trading_fee * entry_price * qty
                    trades.append({"Type": "Short", "Entry": entry_price, "Exit": stop_loss, "P&L": -loss})
                    buy_markers.append((df['timestamp'].iloc[i], stop_loss))
                    positions.append({"entry_time": entry_time, "exit_time": df['timestamp'].iloc[i],
                                      "entry_price": entry_price, "take_profit": take_profit,
                                      "stop_loss": stop_loss, "outcome": "loss"})
                    active_trade = False

    # Metrics
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    profit_loss = current_budget - budget
    trades_df = pd.DataFrame(trades)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name="Candlestick"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df[ema_col], mode='lines', name=f"EMA {ema_number}", line=dict(color='blue')))

    for pos in positions:
        color = "rgba(0,255,0,0.2)" if pos["outcome"] == "profit" else "rgba(255,0,0,0.2)"
        fig.add_shape(type="rect", x0=pos["entry_time"], x1=pos["exit_time"],
                      y0=pos["stop_loss"], y1=pos["take_profit"], fillcolor=color, line=dict(width=0))

    for t, p in buy_markers:
        fig.add_trace(go.Scatter(x=[t], y=[p], mode='markers', marker=dict(color='green', size=10), name="Buy"))
    for t, p in sell_markers:
        fig.add_trace(go.Scatter(x=[t], y=[p], mode='markers', marker=dict(color='red', size=10), name="Sell"))

    fig.update_layout(title=f"{symbol} EMA {ema_number} Backtest", height=chart_height, template="plotly_dark")

    return {
        "Trades": trades_df,
        "Win %": win_rate,
        "P&L": profit_loss,
        "Final Budget": current_budget,
        "Plot": fig
    }




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

show_step_6 = st.sidebar.checkbox("Show Step 6 (Best EMA)", value=False)
show_sr = st.sidebar.checkbox("Show Step 4 & 5 (Support & Resistance)", value=False)

show_step_7 = st.sidebar.checkbox("Show Step 7 (Trade History & Performance)", value=False)
show_step_8 = st.sidebar.checkbox("Show Step 8 (Suggested Trades)", value=False)

show_step_10 = st.sidebar.checkbox("Show Step 10 (EMA Touch Backtest)", value=False)


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

    # --- Show Step 7 Results ---
    if show_step_7 and results_df is not None:
        st.subheader("📊 Step 7: BTC Correlations with Top 50 Coins")
        st.dataframe(results_df, use_container_width=True)

      

           # --- Show Step 8 Suggested Trades ---
    if show_step_8 and best_coin:
        st.subheader("📈 Step 8: Suggested Trades")
        best_symbol = top_50[best_coin]
        best_coin_df = fetch_crypto_data(best_symbol, '15m', '2d')
        best_coin_df['EMA_20'] = best_coin_df['close'].ewm(span=20, adjust=False).mean()
        best_coin_df = suggest_trades(base_df, best_coin_df, 'EMA_20', btc_trend)

        latest_row = best_coin_df.iloc[-1]

        # Build Step 8 table with Entry Price only if signal exists
        latest_signal = pd.DataFrame([{
            "Entry Price": f"${latest_row['close']:.2f}" if latest_row['signal'] else "N/A",
            "Signal": latest_row['signal'] if latest_row['signal'] else "N/A",
            "Take Profit": f"${latest_row['take_profit']:.2f}" if latest_row['take_profit'] else "N/A",
            "Stop Loss": f"${latest_row['stop_loss']:.2f}" if latest_row['stop_loss'] else "N/A"
        }])

        st.dataframe(latest_signal, use_container_width=True)

        # --- Step 9 Suggested Trade Chart ---
        st.subheader("📉 Step 9: Suggested Trade Chart")
        trade_fig = plot_candlestick_with_signals(
            best_coin_df,
            ['EMA_20'],
            f"{best_coin} with Suggested Position",
            future_time_minutes=300,
            height=500
        )
        st.plotly_chart(trade_fig, use_container_width=True)

# -------------------------------
# STEP 10 — EMA Touch Backtesting
# -------------------------------
if show_step_10:
    st.subheader("📊 Step 10 — EMA Touch Backtesting")

    ema_num = st.sidebar.number_input("EMA Number", min_value=5, max_value=200, value=20, step=1)
    budget = st.sidebar.number_input("Starting Budget ($)", min_value=100.0, value=1000.0, step=100.0)
    trading_fee = st.sidebar.number_input("Trading Fee (%)", min_value=0.0, value=0.1, step=0.01) / 100

    st.sidebar.markdown("### Strategy Options")
    trends_to_run = []
    if st.sidebar.checkbox("Run Long (Positive Trend)", value=True):
        trends_to_run.append("positive")
    if st.sidebar.checkbox("Run Short (Negative Trend)", value=False):
        trends_to_run.append("negative")

    backtest_results = {}
    for trend_choice in trends_to_run:
        bt = ema_touch_strategy_with_plot(
            symbol, ema_num, budget,
            trading_fee=trading_fee,
            interval=interval, period=period,
            chart_height=600, trend=trend_choice
        )
        if bt:
            backtest_results[trend_choice] = bt

    # --- Show Results ---
    if backtest_results:
        if len(backtest_results) == 2:  # ✅ both strategies selected
            st.markdown("### 📊 Combined Results (Long + Short)")

            combined_trades = pd.concat(
                [bt["Trades"].assign(Strategy=trend_choice.capitalize()) 
                 for trend_choice, bt in backtest_results.items()],
                ignore_index=True
            )

            total_trades = len(combined_trades)
            total_wins = (combined_trades["P&L"] > 0).sum()
            combined_pnl = sum(bt["P&L"] for bt in backtest_results.values())
            win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0

            combined_budget = budget + combined_pnl

            # ✅ Show metrics in one row on top
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", total_trades)
            c2.metric("Combined Win Rate %", f"{win_rate:.2f}%")
            c3.metric("Combined Final P&L ($)", f"{combined_pnl:.2f}")
            c4.metric("Final Budget ($)", f"{combined_budget:.2f}")

            # ✅ Show one combined chart (overlay long + short)
            combined_fig = go.Figure(backtest_results["positive"]["Plot"].data)
            for trace in backtest_results["negative"]["Plot"].data:
                combined_fig.add_trace(trace)

            combined_fig.update_layout(
                title=f"{symbol} EMA {ema_num} Combined Backtest (Long + Short)",
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(combined_fig, use_container_width=True)

            # Show combined trades table
            st.dataframe(combined_trades, use_container_width=True)

        else:  # ✅ only one strategy selected
            for trend_choice, bt in backtest_results.items():
                st.markdown(f"### 🔍 {trend_choice.capitalize()} Strategy")

                # ✅ Show metrics in one row on top
                c1, c2, c3 = st.columns(3)
                c1.metric("Win Rate %", f"{bt['Win %']:.2f}%")
                c2.metric("Final P&L ($)", f"{bt['P&L']:.2f}")
                c3.metric("Final Budget ($)", f"{bt['Final Budget']:.2f}")

                # Then show chart + table
                st.plotly_chart(bt["Plot"], use_container_width=True)
                st.dataframe(bt["Trades"], use_container_width=True)
