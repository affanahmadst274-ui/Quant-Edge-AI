# pages/3_Crypto_Chatbot.py
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# -------------------------------
# Function to fetch crypto data
# -------------------------------
def fetch_crypto_data(symbol: str, interval: str, period: str):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if not df.empty:
            df.reset_index(inplace=True)
            df.rename(columns={
                "Close": "close", "Open": "open",
                "High": "high", "Low": "low",
                "Volume": "volume"
            }, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

# -------------------------------
# Fundamental Information (Top 20)
# -------------------------------
fundamentals = {
    "BTC": "Bitcoin (BTC) — Launched 2009 by Satoshi Nakamoto. It’s the first and most valuable cryptocurrency, often used as digital gold and a store of value.",
    "ETH": "Ethereum (ETH) — Developed by Vitalik Buterin & team in 2015. A programmable blockchain enabling smart contracts, NFTs, and DeFi.",
    "USDT": "Tether (USDT) — A stablecoin pegged to the US Dollar, commonly used for trading and as a proxy for USD in crypto markets.",
    "XRP": "XRP (Ripple) — Created by Ripple Labs for fast, low-cost cross-border payments and liquidity between fiat currencies.",
    "BNB": "Binance Coin (BNB) — Native token of Binance exchange; used for trading fee discounts and powering the Binance Smart Chain ecosystem.",
    "SOL": "Solana (SOL) — High-speed, low-cost blockchain known for DeFi, NFTs, and scalable Web3 applications.",
    "USDC": "USD Coin (USDC) — Stablecoin fully backed by USD-reserves; highly transparent and regulated.",
    "ADA": "Cardano (ADA) — Founded by Charles Hoskinson; research-driven PoS platform focused on scalability, sustainability, and academia.",
    "DOGE": "Dogecoin (DOGE) — Originated as a meme coin in 2013; now popular for tipping and community-driven culture.",
    "DOT": "Polkadot (DOT) — Enables interoperability between blockchains through parachains, founded by Gavin Wood.",
    "MATIC": "Polygon (MATIC) — Layer-2 scaling solution for Ethereum that offers faster and cheaper transactions.",
    "AVAX": "Avalanche (AVAX) — Scalable smart-contract platform with high throughput and low latency consensus.",
    "SHIB": "Shiba Inu (SHIB) — Meme token with its own DEX (ShibaSwap) and Layer-2 scaling (Shibarium).",
    "LTC": "Litecoin (LTC) — A fork of Bitcoin by Charlie Lee in 2011; offers faster block times and is considered 'silver' to BTC's 'gold'.",
    "LINK": "Chainlink (LINK) — Decentralized oracle network connecting smart contracts with real-world data.",
    "ALGO": "Algorand (ALGO) — Pure PoS blockchain built by Turing Award winner Silvio Micali; aims for scalability and decentralization.",
    "XLM": "Stellar (XLM) — Built for cross-border payments, especially for unbanked regions; co-created by Jed McCaleb.",
    "UNI": "Uniswap (UNI) — Governance token of Uniswap, a leading decentralized exchange using automated liquidity pools.",
    "ETC": "Ethereum Classic (ETC) — Original Ethereum chain post-DAO fork; immutable blockchain preserving original history.",
    "FIL": "Filecoin (FIL) — Decentralized storage network where users rent out spare storage in exchange for FIL tokens."
}

# -------------------------------
# Trade Setup Generator
# -------------------------------
def generate_trade_setup(df, coin):
    if df.empty: 
        return f"Not enough data for {coin}."
    
    df["EMA20"] = df["close"].ewm(span=20).mean()
    support = df["close"].min()
    resistance = df["close"].max()
    last_price = df["close"].iloc[-1]

    entry = round((support + last_price) / 2, 2)
    stoploss = round(support * 0.97, 2)
    target1 = round((entry + resistance) / 2, 2)
    target2 = round(resistance, 2)

    trend = "📈 Bullish" if last_price > df["EMA20"].iloc[-1] else "📉 Bearish"

    return (
        f"**📊 Trade setup for {coin}:**\n\n"
        f"✅ **Entry:** ${entry}\n"
        f"🎯 **Target 1:** ${target1}\n"
        f"🎯 **Target 2:** ${target2}\n"
        f"❌ **Stoploss:** ${stoploss}\n"
        f"📌 **Trend:** {trend}\n\n"
        "⚠️ Educational only, not financial advice."
    )

# -------------------------------
# DCA / Spot Levels Generator
# -------------------------------
def generate_dca_levels(df, coin):
    if df.empty:
        return f"Not enough data for {coin}."
    
    last_price = df["close"].iloc[-1]

    # Simple DCA levels based on percentage discounts
    level1 = round(last_price * 0.90, 2)  # 10% lower
    level2 = round(last_price * 0.80, 2)  # 20% lower
    level3 = round(last_price * 0.70, 2)  # 30% lower

    return (
        f"**💰 Spot DCA Levels for {coin}:**\n\n"
        f"📉 **Level 1:** ${level1}\n"
        f"📉 **Level 2:** ${level2}\n"
        f"📉 **Level 3:** ${level3}\n\n"
        "⚠️ Educational only, not financial advice."
    )

# -------------------------------
# Chatbot Logic with Auto Coin Detection
# -------------------------------
def chatbot_response(user_input: str):
    user_input = user_input.lower()

    supported_coins = list(fundamentals.keys())
    detected = next((coin for coin in supported_coins if coin.lower() in user_input), None)
    symbol = f"{detected}-USD" if detected else "BTC-USD"
    coin = symbol.split("-")[0]

    # Greetings
    if any(greet in user_input for greet in ["hello", "hi"]):
        return f"Hello 👋 Ask me about {coin} or other Top 20 coins’ fundamentals, prices, trends, predictions, trade setups, or DCA levels."

    # Fundamentals
    if any(keyword in user_input for keyword in ["what is", "tell me about", "fundamental", "project"]):
        return fundamentals.get(coin, f"Sorry, I don't have fundamentals info for {coin} yet.")

    # Price
    if "price" in user_input or "current" in user_input:
        df = fetch_crypto_data(symbol, "1h", "1d")
        if not df.empty:
            return f"The latest price of {coin} is **${df['close'].iloc[-1]:.2f}**."
        return f"Could not fetch price for {coin}."

    # Prediction / forecast
    if "prediction" in user_input or "forecast" in user_input:
        df = fetch_crypto_data(symbol, "1h", "7d")
        if not df.empty:
            df["EMA20"] = df["close"].ewm(span=20).mean()
            change = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100
            trend = "📈 bullish" if df["close"].iloc[-1] > df["EMA20"].iloc[-1] else "📉 bearish"
            return (f"Analysis for {coin}: currently {trend}.\n"
                    f"7-day change: {change:.2f}%\n"
                    "⚠️ Not financial advice—just a quick technical insight.")
        return f"Couldn't generate prediction for {coin}."

    # Trend / EMA
    if "trend" in user_input or "ema" in user_input or "moving average" in user_input:
        df = fetch_crypto_data(symbol, "1h", "7d")
        if not df.empty:
            df["EMA20"] = df["close"].ewm(span=20).mean()
            above = df["close"].iloc[-1] > df["EMA20"].iloc[-1]
            return f"{coin} is currently **{'above' if above else 'below'} EMA20**, indicating a {'bullish' if above else 'bearish'} trend."
        return f"Couldn't analyze trend for {coin}."

    # Trade setup
    if "trade setup" in user_input or "setup" in user_input:
        df = fetch_crypto_data(symbol, "1h", "30d")
        return generate_trade_setup(df, coin)

    # DCA / Spot buying
    if "dca" in user_input or "spot" in user_input or "buying" in user_input:
        df = fetch_crypto_data(symbol, "1h", "90d")
        return generate_dca_levels(df, coin)

    return "Not sure I understand — try asking about fundamentals, price, trend, prediction, trade setup, or DCA levels."

# -------------------------------
# Streamlit Chatbot Page
# -------------------------------
st.title("🤖 Crypto Chatbot Assistant")
st.write("Ask about fundamentals, prices, EMA trends, predictions, trade setups, or DCA levels.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your question…")

if user_input:
    response = chatbot_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
