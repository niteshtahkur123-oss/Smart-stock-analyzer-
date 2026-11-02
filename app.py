import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime

st.set_page_config(
    page_title="Smart Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Stock Analyzer")
st.caption("SSA Version 0.3 | NSE Swing Stock Analyzer")

st.warning(
    "Yahoo Finance data may be delayed or unavailable. "
    "Verify prices with your broker. Signals are not guaranteed."
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

if "scan_history" not in st.session_state:
    st.session_state.scan_history = pd.DataFrame()

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

    trading_cost = st.number_input(
        "Estimated Round-trip Cost (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.05
    )

    st.caption(
        "Cost assumption is used only in the backtest."
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
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50)

    return rsi


def download_stock(symbol):
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in required):
        return None

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df if not df.empty else None


def prepare_indicators(df):

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    df = df.copy()

    df["EMA9"] = close.ewm(
        span=9, adjust=False
    ).mean()

    df["EMA20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    df["RSI"] = calculate_rsi(close)

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    macd_signal = macd.ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Bullish"] = macd > macd_signal

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["AvgVolume"] = volume.rolling(20).mean()

    df["VolumeRatio"] = (
        volume / df["AvgVolume"].replace(0, np.nan)
    )

    return df


def analyze_stock(symbol):

    raw = download_stock(symbol)

    if raw is None or len(raw) < 30:
        return None, None, "Insufficient data"

    df = prepare_indicators(raw)

    row = df.iloc[-1]

    price = float(row["Close"])
    rsi = float(row["RSI"])

    if not np.isfinite(rsi):
        return None, df, "RSI unavailable"

    ema9 = float(row["EMA9"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])

    ema200 = (
        float(row["EMA200"])
        if len(df) >= 200 else np.nan
    )

    atr = float(row["ATR"])

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02

    volume_ratio = float(row["VolumeRatio"])

    bullish_macd = bool(row["MACD_Bullish"])

    high52 = float(df["High"].tail(252).max())
    low52 = float(df["Low"].tail(252).min())

    range52 = high52 - low52

    position52 = (
        ((price - low52) / range52) * 100
        if range52 > 0 else 50
    )

    if price > ema20 > ema50:
        trend = "BULLISH"
    elif price < ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    score = 0

    score += int(price > ema20)
    score += int(ema20 > ema50)
    score += int(bullish_macd)
    score += int(50 <= rsi <= 70)
    score += int(
        np.isfinite(volume_ratio) and volume_ratio >= 1.5
    )
    score += int(
        np.isfinite(ema200) and price > ema200
    )

    if score >= 5:
        signal = "BUY WATCH"
        strength = "STRONG"
    elif score >= 3:
        signal = "WATCH"
        strength = "MEDIUM"
    else:
        signal = "WAIT"
        strength = "WEAK"

    entry = price
    stop = max(0.01, entry - 1.5 * atr)
    target1 = entry + 2 * atr
    target2 = entry + 3 * atr

    risk = entry - stop

    rr1 = (target1 - entry) / risk if risk > 0 else 0
    rr2 = (target2 - entry) / risk if risk > 0 else 0

    result = {
        "Stock": symbol.replace(".NS", ""),
        "Price ₹": round(price, 2),
        "Trend": trend,
        "52W Position %": round(position52, 1),
        "52W Low ₹": round(low52, 2),
        "52W High ₹": round(high52, 2),
        "RSI": round(rsi, 1),
        "EMA 9": round(ema9, 2),
        "EMA 20": round(ema20, 2),
        "EMA 50": round(ema50, 2),
        "EMA 200": round(ema200, 2)
        if np.isfinite(ema200) else None,
        "Volume Ratio": round(volume_ratio, 2)
        if np.isfinite(volume_ratio) else None,
        "Entry Ref ₹": round(entry, 2),
        "Stop Loss Ref ₹": round(stop, 2),
        "Target 1 ₹": round(target1, 2),
        "Target 2 ₹": round(target2, 2),
        "R:R Target 1": round(rr1, 2),
        "R:R Target 2": round(rr2, 2),
        "Score": score,
        "Strength": strength,
        "Signal": signal,
        "Data Date": str(df.index[-1].date())
    }

    return result, df, "OK"


def backtest(df, cost_percent):

    if len(df) < 100:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_price = df["Open"]

    trades = []

    cost = cost_percent / 100

    # Signal is generated after a completed daily candle.
    # Entry is at the following day's open.
    for i in range(60, len(df) - 1):

        row = df.iloc[i]

        price = float(row["Close"])
        rsi = float(row["RSI"])

        if not np.isfinite(rsi):
            continue

        score = 0

        score += int(price > row["EMA20"])
        score += int(row["EMA20"] > row["EMA50"])
        score += int(row["MACD_Bullish"])
        score += int(50 <= rsi <= 70)
        score += int(
            np.isfinite(row["VolumeRatio"])
            and row["VolumeRatio"] >= 1.5
        )
        score += int(
            i >= 199 and price > row["EMA200"]
        )

        if score < 5:
            continue

        entry = float(open_price.iloc[i + 1])

        atr = float(row["ATR"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        stop = entry - 1.5 * atr
        target = entry + 2 * atr

        exit_price = None
        exit_reason = "10-day exit"

        last_day = min(i + 11, len(df))

        for j in range(i + 1, last_day):

            day_low = float(low.iloc[j])
            day_high = float(high.iloc[j])

            # Conservative assumption:
            # If both stop and target are touched,
            # assume stop was hit first.
            if day_low <= stop:
                exit_price = stop
                exit_reason = "Stop Loss"
                break

            if day_high >= target:
                exit_price = target
                exit_reason = "Target"
                break

        if exit_price is None:
            exit_price = float(close.iloc[last_day - 1])

        net_return = (
            (exit_price - entry) / entry
        ) * 100 - cost_percent

        trades.append({
            "Return %": net_return,
            "Exit": exit_reason
        })

    if not trades:
        return {
            "Trades": 0,
            "Win Rate %": 0,
            "Avg Return %": 0
        }

    trade_df = pd.DataFrame(trades)

    return {
        "Trades": len(trade_df),
        "Win Rate %": round(
            (trade_df["Return %"] > 0).mean() * 100, 1
        ),
        "Avg Return %": round(
            trade_df["Return %"].mean(), 2
        )
    }


if st.button("SCAN STOCKS", type="primary"):

    if min_price >= max_price:
        st.error("Maximum price must be greater than minimum price.")

    elif not symbols:
        st.error("Please enter at least one NSE symbol.")

    else:

        results = []
        errors = []
        backtest_results = []

        progress = st.progress(0)
        message = st.empty()

        for i, symbol in enumerate(symbols):

            message.write(
                f"Scanning {symbol} ({i+1}/{len(symbols)})"
            )

            try:
                result, stock_df, status = analyze_stock(symbol)

                if result is not None:

                    if min_price <= result["Price ₹"] <= max_price:
                        results.append(result)

                        bt = backtest(
                            stock_df,
                            trading_cost
                        )

                        if bt is not None:
                            bt["Stock"] = result["Stock"]
                            backtest_results.append(bt)

                else:
                    errors.append({
                        "Stock": symbol,
                        "Reason": status
                    })

            except Exception as e:
                errors.append({
                    "Stock": symbol,
                    "Reason": str(e)[:100]
                })

            progress.progress((i + 1) / len(symbols))

        progress.empty()
        message.empty()

        if results:

            df_result = pd.DataFrame(results)

            df_result = df_result.sort_values(
                ["Score", "Volume Ratio"],
                ascending=False,
                na_position="last"
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

            history = df_result.copy()
            history.insert(
                0,
                "Scan Time",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            st.session_state.scan_history = pd.concat(
                [
                    st.session_state.scan_history,
                    history
                ],
                ignore_index=True
            )

            if backtest_results:

                st.subheader("Historical Backtesting")

                bt_df = pd.DataFrame(backtest_results)

                st.dataframe(
                    bt_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Backtest uses historical daily candles, "
                    "next-day open entry, 2 ATR target, "
                    "1.5 ATR stop and maximum 10-day holding. "
                    "If stop and target are both touched on one candle, "
                    "stop is assumed first. Results are not a prediction."
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
            "daily reference levels. Backtesting is historical and "
            "does not guarantee future performance. "
            "Daily Swing screening only; no 5-minute Intraday signals."
        )


st.divider()

st.subheader("📁 Scan History")

if not st.session_state.scan_history.empty:

    st.dataframe(
        st.session_state.scan_history,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Scan History",
        st.session_state.scan_history.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="SSA_Scan_History.csv",
        mime="text/csv"
    )

else:
    st.write("No scan history in this session yet.")

st.caption(
    "SSA v0.3 | Daily Technical Screening | Manual Trading Only"
)
