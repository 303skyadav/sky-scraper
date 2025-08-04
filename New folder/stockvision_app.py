import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Page config ---
st.set_page_config(page_title="StockVision AI", layout="centered")
st.title("📈 StockVision AI – Candlestick Chart with Full Controls")

# --- Load Top 500 NSE Stocks ---
@st.cache_data
def load_symbols():
    df = pd.read_csv("ind_nifty500list.csv")
    df['Symbol'] = df['Symbol'].astype(str).str.strip() + '.NS'
    return sorted(df['Symbol'].tolist())

symbols = load_symbols()
selected = st.selectbox("Select a stock:", symbols)

# --- Fetch Historical Data (Daily, 5 years) ---
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    return yf.Ticker(symbol).history(period="10y", interval="1d")

df = fetch_data(selected)

# --- Plot Candlestick Chart with Buttons ---
st.subheader("🕯️ Candlestick Chart (Red/Green) + Zoom Controls")

fig1 = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='green',
    decreasing_line_color='red',
    increasing_fillcolor='green',
    decreasing_fillcolor='red',
    name='Candlestick'
)])

fig1.update_layout(
    title=f"{selected} – Daily Candlestick (10Y History)",
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    template="plotly_dark",
    height=700,
    hovermode='x unified',
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="Today", step="day", stepmode="backward"),
                dict(count=7, label="Week", step="day", stepmode="backward"),
                dict(count=1, label="Month", step="month", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="Full")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

st.plotly_chart(fig1, use_container_width=True)

# --- Manual Hard Refresh ---
if st.button("🔁 Hard Refresh"):
    st.rerun()
