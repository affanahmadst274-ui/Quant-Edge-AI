# -------------------------------
# CRYPTO CHATBOT PAGE
# -------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

def crypto_chatbot():
    st.title("🤖 Crypto Chatbot")

    st.markdown("Ask me anything about crypto prices, charts, or analysis!")

    # --- Keep chat history in session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Show chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Input box ---
    if prompt := st.chat_input("Type your crypto question..."):
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Simple crypto Q&A logic ---
        response = handle_crypto_question(prompt)

        # Save bot reply
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


def handle_crypto_question(prompt: str) -> str:
    """
    Simple logic for answering crypto-related questions.
    Extend with LLM API or knowledge base if needed.
    """
    prompt = prompt.lower()

    # Example: "price of btc" or "what is eth price"
    if "price" in prompt:
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]:
            if symbol.split("-")[0].lower() in prompt:
                try:
                    price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
                    return f"💰 The latest price of **{symbol}** is **${price:.2f}**."
                except Exception:
                    return "⚠️ I couldn't fetch the price right now. Try again later."

    # Example: "chart of btc" or "show eth chart"
    if "chart" in prompt:
        for symbol in ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]:
            if symbol.split("-")[0].lower() in prompt:
                try:
                    df = yf.Ticker(symbol).history(period="7d", interval="1h")
                    import plotly.graph_objs as go
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df["Open"], high=df["High"],
                        low=df["Low"], close=df["Close"], name="Candles"
                    ))
                    fig.update_layout(title=f"{symbol} Last 7 Days", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                    return f"📊 Here's the recent chart for **{symbol}**."
                except Exception:
                    return "⚠️ I couldn't fetch the chart right now."

    # Example: fallback
    return "🤔 I can answer questions like 'price of BTC' or 'show ETH chart'. Try asking in that format!"


# ✅ Call on Streamlit page
if __name__ == "__main__":
    crypto_chatbot()
