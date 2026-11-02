import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(page_title="Smart Stock Analyzer", page_icon="📊", layout="wide")

st.title("📊 Smart Stock Analyzer (SSA)")
st.caption("NSE stock monitoring prototype • Research only • No automatic trading")

st.warning(
    "Data note: This prototype uses Yahoo Finance via yfinance. Data may be delayed, incomplete, "
    "or unavailable. It is not an exchange-licensed live feed. Verify prices with your broker before acting."
)

with st.sidebar:
    st.header("Scanner settings")
    min_price = st.number_input("Minimum price (₹)", min_value=1.0, value=100.0, step=5.0)
    max_price = st.number_input("Maximum price (₹)", min_value=min_price, value=200.0, step=5.0)
    max_stocks = st.number_input("Maximum stocks to show", min_value=1, max_value=10, value=10)
    period = st.selectbox("Historical data period", ["1y", "2y", "5y"], index=0)
    st.caption("Edit the NSE symbols below. Add .NS after each symbol.")

default_symbols = """ITC.NS
NHPC.NS
BEL.NS
BANKBARODA.NS
CANBK.NS
PFC.NS
RECLTD.NS
IOC.NS
GAIL.NS
ONGC.NS
TATAPOWER.NS
IRFC.NS
IDFCFIRSTB.NS
SAIL.NS
NMDC.NS
COALINDIA.NS
"""

symbols_text = st.text_area("NSE symbols to scan", value=default_symbols, height=230)
symbols = list(dict.fromkeys(
    s.strip().upper() for s in symbols_text.replace(",", "\n").splitlines() if s.strip()
))
symbols = [s if s.endswith(".NS") else s + ".NS" for s in symbols]

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze_symbol(symbol, period):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty or len(df) < 60:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    if close.empty:
        return None
    last = float(close.iloc[-1])
    high52 = float(df["High"].tail(252).max())
    low52 = float(df["Low"].tail(252).min())
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else np.nan
    rsi14 = float(rsi(close).iloc[-1])
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_bull = bool(macd_line.iloc[-1] > signal_line.iloc[-1])
    avg_vol = float(df["Volume"].tail(20).mean())
    today_vol = float(df["Volume"].iloc[-1])
    vol_ratio = today_vol / avg_vol if avg_vol > 0 else np.nan
    score = 0
    score += 1 if last > ema20 else 0
    score += 1 if ema20 > ema50 else 0
    score += 1 if macd_bull else 0
    score += 1 if 50 <= rsi14 <= 70 else 0
    score += 1 if vol_ratio >= 1.5 else 0
    signal = "BUY WATCH" if score >= 4 else ("WATCH" if score == 3 else "WAIT")
    return {
        "Symbol": symbol.replace(".NS", ""),
        "Latest ₹": round(last, 2),
        "52W Low ₹": round(low52, 2),
        "52W High ₹": round(high52, 2),
        "RSI (14)": round(rsi14, 1),
        "EMA 20 ₹": round(ema20, 2),
        "EMA 50 ₹": round(ema50, 2),
        "EMA 200 ₹": round(ema200, 2) if pd.notna(ema200) else None,
        "Volume x Avg": round(vol_ratio, 2) if pd.notna(vol_ratio) else None,
        "MACD Bullish": "Yes" if macd_bull else "No",
        "Score / 5": score,
        "Status": signal,
        "Data date": str(df.index[-1].date())
    }

if st.button("🔎 Scan stocks", type="primary", use_container_width=True):
    if not symbols:
        st.error("Please enter at least one NSE symbol.")
    else:
        results = []
        progress = st.progress(0)
        status = st.empty()
        for i, symbol in enumerate(symbols):
            status.write(f"Checking {symbol} ({i+1}/{len(symbols)})…")
            try:
                item = analyze_symbol(symbol, period)
                if item and min_price <= item["Latest ₹"] <= max_price:
                    results.append(item)
            except Exception:
                pass
            progress.progress((i + 1) / len(symbols))
        status.empty()
        progress.empty()
        if results:
            result_df = pd.DataFrame(results).sort_values(
                ["Score / 5", "Volume x Avg"], ascending=[False, False]
            ).head(int(max_stocks))
            st.subheader(f"Stocks in ₹{min_price:.0f}–₹{max_price:.0f} range")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scan as CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ssa_scan_{date.today().isoformat()}.csv",
                mime="text/csv"
            )
            st.info(
                "BUY WATCH is only a technical screening flag, not a trade instruction. "
                "This first version uses daily candles; intraday 5-minute signals are not enabled."
            )
        else:
            st.warning("No stocks matched the price range, or data could not be retrieved. Try another symbol list.")

st.divider()
st.subheader("How to use")
st.markdown(
    "1. Keep `.NS` after NSE symbols.  2. Press **Scan stocks**.  "
    "3. Review the latest data date and indicators.  4. Confirm all prices and levels independently "
    "in your broker app before any trade."
)
st.caption("Version 0.1 • Daily-data prototype • Swing screening first")
