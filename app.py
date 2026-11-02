import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date

st.set_page_config(
    page_title="Smart Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Stock Analyzer")
st.caption("SSA Version 0.2 | NSE Stock Monitoring")

st.warning(
    "Yahoo Finance data may be delayed or unavailable. "
    "Verify all prices with your broker. Signals are not guaranteed."
)

DEFAULT_STOCKS = """
ADANIPOWER.NS
ABCAPITAL.NS
ASHOKLEY.NS
BANKBARODA.NS
BANKINDIA.NS
BANDHANBNK.NS
BHEL.NS
BEL.NS
CANBK.NS
CENTRALBK.NS
COALINDIA.NS
CESC.NS
FEDERALBNK.NS
GAIL.NS
HINDPETRO.NS
HUDCO.NS
IDFCFIRSTB.NS
IDBI.NS
IOC.NS
IRB.NS
IRCON.NS
IRCTC.NS
IREDA.NS
IRFC.NS
ITC.NS
JINDALSTEL.NS
JIOFIN.NS
LICHSGFIN.NS
LTF.NS
MANAPPURAM.NS
MOTHERSON.NS
NATIONALUM.NS
NBCC.NS
NHPC.NS
NMDC.NS
NTPC.NS
ONGC.NS
PFC.NS
PNB.NS
POWERGRID.NS
RBLBANK.NS
RECLTD.NS
SAIL.NS
SJVN.NS
SUZLON.NS
TATAPOWER.NS
UNIONBANK.NS
YESBANK.NS
"""

with st.sidebar:
    st.header("Scanner Settings")

    min_price = st.number_input(
        "Minimum Price ₹", value=100.0
    )

    max_price = st.number_input(
        "Maximum Price ₹", value=200.0
    )

    max_stocks = st.slider(
        "Maximum Stocks", 1, 10, 10
    )

    period = st.selectbox(
        "Historical Data", ["1y", "2y", "5y"]
    )

symbols_text = st.text_area(
    "NSE Symbols to Scan",
    DEFAULT_STOCKS,
    height=300
)

symbols = list(dict.fromkeys(
    x.strip().upper()
    for x in symbols_text.replace(",", "\n").splitlines()
    if x.strip()
))

symbols = [
    x if x.endswith(".NS") else x + ".NS"
    for x in symbols
]


def calculate_rsi(close):
    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def analyze_stock(symbol):

    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if df.empty:
        return None, "No data"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()
    volume = df["Volume"].fillna(0)

    if len(close) < 30:
        return None, "Insufficient history"

    price = float(close.iloc[-1])

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    rsi = float(calculate_rsi(close).iloc[-1])

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    macd_signal = macd.ewm(span=9, adjust=False).mean()

    bullish_macd = macd.iloc[-1] > macd_signal.iloc[-1]

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = float(tr.rolling(14).mean().iloc[-1])

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02

    avg_volume = float(volume.tail(20).mean())

    volume_ratio = (
        float(volume.iloc[-1] / avg_volume)
        if avg_volume > 0 else np.nan
    )

    high52 = float(high.tail(252).max())
    low52 = float(low.tail(252).min())

    score = 0

    score += int(price > ema20.iloc[-1])
    score += int(ema20.iloc[-1] > ema50.iloc[-1])
    score += int(bullish_macd)
    score += int(50 <= rsi <= 70)
    score += int(np.isfinite(volume_ratio) and volume_ratio >= 1.5)
    score += int(
        len(close) >= 200 and price > ema200.iloc[-1]
    )

    if score >= 5:
        signal = "BUY WATCH"
    elif score >= 3:
        signal = "WATCH"
    else:
        signal = "WAIT"

    entry = price
    stop = max(0.01, entry - 1.5 * atr)
    target1 = entry + 2 * atr
    target2 = entry + 3 * atr

    return {
        "Stock": symbol.replace(".NS", ""),
        "Price ₹": round(price, 2),
        "52W Low ₹": round(low52, 2),
        "52W High ₹": round(high52, 2),
        "RSI": round(rsi, 1),
        "EMA 9": round(float(ema9.iloc[-1]), 2),
        "EMA 20": round(float(ema20.iloc[-1]), 2),
        "EMA 50": round(float(ema50.iloc[-1]), 2),
        "EMA 200": round(float(ema200.iloc[-1]), 2)
        if len(close) >= 200 else None,
        "Volume Ratio": round(volume_ratio, 2)
        if np.isfinite(volume_ratio) else None,
        "Entry Ref ₹": round(entry, 2),
        "Stop Loss Ref ₹": round(stop, 2),
        "Target 1 ₹": round(target1, 2),
        "Target 2 ₹": round(target2, 2),
        "Score": score,
        "Signal": signal,
        "Data Date": str(df.index[-1].date())
    }, "OK"


if st.button("SCAN STOCKS", type="primary"):

    results = []
    errors = []

    progress = st.progress(0)
    message = st.empty()

    for i, symbol in enumerate(symbols):

        message.write(
            f"Scanning {symbol} ({i+1}/{len(symbols)})"
        )

        try:
            result, status = analyze_stock(symbol)

            if result is not None:
                if min_price <= result["Price ₹"] <= max_price:
                    results.append(result)
            else:
                errors.append({
                    "Stock": symbol,
                    "Reason": status
                })

        except Exception:
            errors.append({
                "Stock": symbol,
                "Reason": "Data fetch error"
            })

        progress.progress((i + 1) / len(symbols))

    progress.empty()
    message.empty()

    if results:

        df_result = pd.DataFrame(results)

        df_result = df_result.sort_values(
            "Score",
            ascending=False
        ).head(max_stocks)

        st.subheader(
            f"Qualifying Stocks: {len(df_result)}"
        )

        st.dataframe(
            df_result,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Scan CSV",
            df_result.to_csv(index=False).encode("utf-8"),
            file_name=f"SSA_{date.today()}.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "No stocks matched the price range or data was unavailable."
        )

    with st.expander("Data Fetch Diagnostics"):

        if errors:
            st.dataframe(
                pd.DataFrame(errors),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("No data errors recorded.")

    st.info(
        "Entry, Stop Loss and Targets are illustrative ATR-based "
        "daily reference levels, not validated trade recommendations. "
        "This version supports daily Swing screening only, not "
        "5-minute Intraday signals."
    )

st.divider()

st.caption(
    "SSA v0.2 | Daily Technical Screening | Manual Trading Only"
)
