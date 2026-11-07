import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(
    page_title="Smart Stock Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Stock Analyzer")
"SSA v0.5 | Portfolio Backtesting | Manual Trading Only"
st.warning(
    "Yahoo Finance data may be delayed or unavailable. "
    "Verify prices with your broker. Historical results "
    "do not guarantee future performance."
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
        "Historical Data",
        ["1y", "2y", "5y"],
        index=1
    )

    trading_cost = st.number_input(
        "Round-trip Trading Cost %",
        min_value=0.0,
        max_value=2.0,
        value=0.20,
        step=0.05
    )

    holding_days = st.slider(
        "Maximum Holding Days",
        5, 20, 10
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
# ==========================================
# SSA LIVE MARKET MONITOR
# ==========================================

st.divider()
st.header("📡 Live Market Monitor")

st.caption(
    "Source: Yahoo Finance | "
    "Auto Refresh: 10 Seconds | "
    "Indicative Data Only"
)


@st.fragment(run_every="10s")
def live_market_monitor():

    st.caption(
        "Last Screen Update: "
        + pd.Timestamp.now(tz="Asia/Kolkata").strftime(
            "%d-%m-%Y %H:%M:%S"
        )
    )

    try:

        live_data = yf.download(
            tickers=symbols,
            period="5d",
            interval="5m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30
        )

        if live_data is None or live_data.empty:
            st.error("DATA DISCONNECTED")
            return

        records = []
        stock_frames = {}

        for symbol in symbols:

            try:

                if isinstance(live_data.columns, pd.MultiIndex):

                    if symbol in live_data.columns.get_level_values(0):
                        stock_df = live_data[symbol].copy()

                    elif symbol in live_data.columns.get_level_values(1):
                        stock_df = live_data.xs(
                            symbol,
                            axis=1,
                            level=1
                        ).copy()

                    else:
                        continue

                else:

                    if len(symbols) != 1:
                        continue

                    stock_df = live_data.copy()

                if isinstance(stock_df.columns, pd.MultiIndex):
                    stock_df.columns = (
                        stock_df.columns.get_level_values(0)
                    )

                if "Close" not in stock_df.columns:
                    continue

                stock_df = stock_df.dropna(
                    subset=["Close"]
                )

                if stock_df.empty:
                    continue

                # Convert timestamps to Indian Standard Time
                if stock_df.index.tz is None:
                    stock_df.index = stock_df.index.tz_localize(
                        "Asia/Kolkata"
                    )
                else:
                    stock_df.index = stock_df.index.tz_convert(
                        "Asia/Kolkata"
                    )

                stock_frames[symbol] = stock_df.copy()

                current_price = float(
                    stock_df["Close"].iloc[-1]
                )

                timestamp = stock_df.index[-1]

                dates = stock_df.index.date

                unique_dates = list(dict.fromkeys(dates))

                previous_close = np.nan

                if len(unique_dates) >= 2:

                    previous_day = unique_dates[-2]

                    previous_rows = stock_df[
                        dates == previous_day
                    ]

                    if not previous_rows.empty:
                        previous_close = float(
                            previous_rows["Close"].iloc[-1]
                        )

                change = (
                    current_price - previous_close
                    if np.isfinite(previous_close)
                    else np.nan
                )

                change_pct = (
                    change / previous_close * 100
                    if np.isfinite(previous_close)
                    and previous_close > 0
                    else np.nan
                )

                volume = (
                    float(stock_df["Volume"].iloc[-1])
                    if "Volume" in stock_df.columns
                    else np.nan
                )

                age_seconds = (
                    pd.Timestamp.now(
                        tz="Asia/Kolkata"
                    ) - timestamp
                ).total_seconds()

                status = (
                    "FRESH"
                    if 0 <= age_seconds <= 600
                    else "STALE"
                )

                records.append({

                    "Stock": symbol.replace(".NS", ""),
                    "LTP ₹": round(current_price, 2),

                    "Change ₹": (
                        round(change, 2)
                        if np.isfinite(change)
                        else None
                    ),

                    "Change %": (
                        round(change_pct, 2)
                        if np.isfinite(change_pct)
                        else None
                    ),

                    "Last 5m Volume": (
                        int(volume)
                        if np.isfinite(volume)
                        else None
                    ),

                    "Last Data Time": timestamp.strftime(
                        "%H:%M:%S"
                    ),

                    "Status": status
                })

            except Exception:
                continue

        if not records:

            st.warning("No usable market data received.")
            return

        # ======================================
        # LIVE MARKET TABLE
        # ======================================

        monitor_df = pd.DataFrame(records)

        st.metric(
            "Stocks Received",
            f"{len(monitor_df)} / {len(symbols)}"
        )

        st.dataframe(
            monitor_df,
            use_container_width=True,
            hide_index=True
        )

        fresh_count = (
            monitor_df["Status"] == "FRESH"
        ).sum()

        stale_count = (
            monitor_df["Status"] == "STALE"
        ).sum()

        c1, c2 = st.columns(2)

        c1.metric("Fresh Data", int(fresh_count))
        c2.metric("Stale Data", int(stale_count))

        # ======================================
        # SELECT STOCK FOR LIVE CHART
        # ======================================

        st.divider()

        st.subheader("📈 LIVE 5-MINUTE CHART")

        stock_names = [
            s.replace(".NS", "")
            for s in stock_frames.keys()
        ]

        if not stock_names:
            st.warning("No chart data available.")
            return

        selected_stock = st.selectbox(
            "Select Stock",
            stock_names,
            key="live_5m_chart_stock"
        )

        selected_symbol = selected_stock + ".NS"

        chart_df = stock_frames[selected_symbol].copy()

        # Convert chart timestamps to IST
        if chart_df.index.tz is None:
            chart_df.index = chart_df.index.tz_localize(
                "Asia/Kolkata"
            )
        else:
            chart_df.index = chart_df.index.tz_convert(
                "Asia/Kolkata"
            )

        # Preserve IST clock time for Plotly
        chart_df.index = chart_df.index.tz_localize(None)

        chart_df = chart_df.sort_index()

        required = ["Open", "High", "Low", "Close"]

        if not all(c in chart_df.columns for c in required):

            st.warning("OHLC chart data unavailable.")

        else:

            # EMA calculation on full 5-day data
            chart_df["EMA9"] = (
                chart_df["Close"].ewm(
                    span=9,
                    adjust=False
                ).mean()
            )

            chart_df["EMA20"] = (
                chart_df["Close"].ewm(
                    span=20,
                    adjust=False
                ).mean()
            )

            # BUY crossover
            chart_df["BUY"] = (
                (chart_df["EMA9"] > chart_df["EMA20"]) &
                (chart_df["EMA9"].shift(1) <=
                 chart_df["EMA20"].shift(1))
            )

            # SELL crossover
            chart_df["SELL"] = (
                (chart_df["EMA9"] < chart_df["EMA20"]) &
                (chart_df["EMA9"].shift(1) >=
                 chart_df["EMA20"].shift(1))
            )

            # Filter only after calculating EMA and signals
            latest_date = chart_df.index[-1].date()

            chart_df = chart_df[
                chart_df.index.date == latest_date
            ].copy()

            # ==================================
            # BUY / SELL CONDITIONS
            # ==================================

            chart_df["BUY"] = (
                (chart_df["EMA9"] > chart_df["EMA20"]) &
                (chart_df["EMA9"].shift(1) <=
                 chart_df["EMA20"].shift(1))
            )

            chart_df["SELL"] = (
                (chart_df["EMA9"] < chart_df["EMA20"]) &
                (chart_df["EMA9"].shift(1) >=
                 chart_df["EMA20"].shift(1))
            )

            # Keep full data for EMA calculation,
            # but display only latest trading day
            latest_date = chart_df.index[-1].date()

            chart_df = chart_df[
                chart_df.index.date == latest_date
            ] 

            # ==================================
            # LATEST SIGNAL
            # ==================================

            latest = chart_df.iloc[-1]

            if bool(latest["BUY"]):

                st.success("🟢 BUY SIGNAL")

            elif bool(latest["SELL"]):

                st.error("🔴 SELL SIGNAL")

            elif latest["EMA9"] > latest["EMA20"]:

                st.info("EMA TREND: BULLISH")

            else:

                st.warning("EMA TREND: BEARISH")

            # ==================================
            # CANDLESTICK CHART
            # ==================================

            fig = go.Figure()

            fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="Price",
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
        increasing_fillcolor="#16A34A",
        decreasing_fillcolor="#DC2626"
    )
)

            # BUY SIGNAL MARKERS
            buy_points = chart_df[chart_df["BUY"]]

            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["Low"] * 0.998,
                    mode="markers+text",
                    name="BUY",
                    text=["BUY"] * len(buy_points),
                    textposition="bottom center",
                    marker=dict(
                        symbol="triangle-up",
                        size=16,
                        color="green",
                        line=dict(width=2, color="white")
                    )
                )
            )

            # SELL SIGNAL MARKERS
            sell_points = chart_df[chart_df["SELL"]]

            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["High"] * 1.002,
                    mode="markers+text",
                    name="SELL",
                    text=["SELL"] * len(sell_points),
                    textposition="top center",
                    marker=dict(
                        symbol="triangle-down",
                        size=16,
                        color="red",
                        line=dict(width=2, color="white")
                    )
                )
            )            

            # EMA 9

            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df["EMA9"],
                    mode="lines",
                    name="EMA 9"
                )
            )

            # EMA 20

            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=chart_df["EMA20"],
                    mode="lines",
                    name="EMA 20"
                )
            )

            # BUY MARKERS

            buy_points = chart_df[
                chart_df["BUY"]
            ]

            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["Low"] * 0.998,
                    mode="markers+text",
                    name="BUY",
                    text=["BUY"] * len(buy_points),
                    textposition="bottom center"
                )
            )

            # SELL MARKERS

            sell_points = chart_df[
                chart_df["SELL"]
            ]

            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["High"] * 1.002,
                    mode="markers+text",
                    name="SELL",
                    text=["SELL"] * len(sell_points),
                    textposition="top center"
                )
            )

            # ==================================
            # CHART LAYOUT
            # ==================================

            fig.update_layout(
                title=f"{selected_stock} | {latest_date.strftime('%d-%b-%Y')}",
                xaxis_title="Time",
                yaxis_title="Price (₹)",
                height=650,
                hovermode="x unified",
                dragmode="pan",
                xaxis_rangeslider_visible=False
            )

            fig.update_xaxes(
                type="date",
                tickformat="%H:%M",
                tick0=f"{latest_date} 09:15:00",
                dtick=600000,
                tickangle=0,
                rangebreaks=[
                    dict(bounds=[15.5, 9.25], pattern="hour")
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "responsive": True,
                    "modeBarButtonsToAdd": [
                        "zoomIn2d",
                        "zoomOut2d",
                        "resetScale2d"
                    ]
                }
            )

    except Exception as e:

        st.error(
            f"Market Data Error: {str(e)[:200]}"
        )


live_market_monitor()

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

    required = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    if not all(c in df.columns for c in required):
        return None

    df = df[required].copy()

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return df if len(df) else None


def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    for span in [9, 20, 50, 200]:
        df[f"EMA{span}"] = close.ewm(
            span=span,
            adjust=False
        ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    loss = (-delta.clip(upper=0)).ewm(
        alpha=1/14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    df.loc[
        (loss == 0) & (gain > 0), "RSI"
    ] = 100

    df.loc[
        (loss == 0) & (gain == 0), "RSI"
    ] = 50

    macd = (
        close.ewm(span=12, adjust=False).mean()
        - close.ewm(span=26, adjust=False).mean()
    )

    macd_signal = macd.ewm(
        span=9,
        adjust=False
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


def get_score(row, index):

    rsi = row["RSI"]

    if not np.isfinite(rsi):
        return 0

    score = 0

    score += int(row["Close"] > row["EMA20"])
    score += int(row["EMA20"] > row["EMA50"])
    score += int(row["MACD_Bullish"])
    score += int(50 <= rsi <= 70)

    score += int(
        np.isfinite(row["VolumeRatio"])
        and row["VolumeRatio"] >= 1.5
    )

    score += int(
        index >= 199
        and row["Close"] > row["EMA200"]
    )

    return score


def analyze_stock(symbol, df):

    if df is None or len(df) < 30:
        return None

    df = calculate_indicators(df)

    row = df.iloc[-1]

    price = float(row["Close"])
    rsi = float(row["RSI"])

    if not np.isfinite(rsi):
        return None

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

    high52 = float(df["High"].tail(252).max())
    low52 = float(df["Low"].tail(252).min())

    position52 = (
        (price - low52) / (high52 - low52) * 100
        if high52 > low52 else 50
    )

    if price > ema20 > ema50:
        trend = "BULLISH"
    elif price < ema20 < ema50:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    score = get_score(row, len(df) - 1)

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

    return {
        "Stock": symbol.replace(".NS", ""),
        "Price ₹": round(price, 2),
        "Trend": trend,
        "52W Position %": round(position52, 1),
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


    # Add the newly opened position
    open_positions.append(position)

    # Process remaining open positions after all entry signals
    for pos in open_positions:
        cash += pos["Invested ₹"] + pos["P&L ₹"]
        equity += pos["P&L ₹"]
        completed.append(pos)

    open_positions = []

    if not completed:
        return [], {}

    results = pd.DataFrame(completed)

    total_trades = len(results)
    winning_trades = int((results["P&L ₹"] > 0).sum())
    losing_trades = int((results["P&L ₹"] < 0).sum())

    total_pnl = float(results["P&L ₹"].sum())
    final_capital = initial_capital + total_pnl

    returns = results["Return %"].astype(float)

    win_rate = (
        winning_trades / total_trades * 100
        if total_trades else 0
    )

    summary = {
        "Initial Capital ₹": round(initial_capital, 2),
        "Final Capital ₹": round(final_capital, 2),
        "Net P&L ₹": round(total_pnl, 2),
        "Total Return %": round(
            (total_pnl / initial_capital) * 100, 2
        ),
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate %": round(win_rate, 2),
        "Average Trade Return %": round(
            float(returns.mean()), 2
        ),
        "Best Trade %": round(float(returns.max()), 2),
        "Worst Trade %": round(float(returns.min()), 2)
    }

    return completed, summary

def calculate_metrics(trades):

    if not trades:
        return None

    tdf = pd.DataFrame(trades)

    returns = tdf["Return %"].astype(float)

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    equity = (1 + returns / 100).cumprod()

    peak = equity.cummax()

    drawdown = (
        (equity / peak) - 1
    ) * 100

    return {
        "Trades": len(tdf),
        "Win Rate %": round(
            (returns > 0).mean() * 100, 1
        ),
        "Profit Factor": round(profit_factor, 2)
        if np.isfinite(profit_factor) else "No Losses",
        "Average Return %": round(
            returns.mean(), 2
        ),
        "Average Win %": round(
            wins.mean(), 2
        ) if len(wins) else 0,
        "Average Loss %": round(
            losses.mean(), 2
        ) if len(losses) else 0,
        "Max Drawdown %": round(
            drawdown.min(), 2
        ),
        "Total Return %": round(
            (equity.iloc[-1] - 1) * 100, 2
        )
    }
    
def run_backtest(df, cost, max_days, symbol=""):

    trades = []

    if len(df) < 220:
        return trades

    i = 199

    while i < len(df) - 1:

        row = df.iloc[i]
        score = get_score(row, i)

        if score < 5:
            i += 1
            continue

        atr = float(row["ATR"])

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        # Entry on next trading day's Open.
        entry_index = i + 1
        entry = float(df["Open"].iloc[entry_index])

        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue

        stop = entry - (1.5 * atr)
        target1 = entry + atr
        target2 = entry + (2 * atr)

        last_index = min(
            entry_index + max_days - 1,
            len(df) - 1
        )

        remaining = 100.0
        booked = 0.0
        weighted_exit = 0.0

        t1_hit = False
        exit_reason = "Time Exit"
        exit_index = last_index

        for j in range(entry_index, last_index + 1):

            day_low = float(df["Low"].iloc[j])
            day_high = float(df["High"].iloc[j])
            day_open = float(df["Open"].iloc[j])

            if not t1_hit:

                # Stop Loss gets priority.
                if day_low <= stop:

                    fill_price = (
                        day_open if day_open < stop else stop
                    )

                    weighted_exit += remaining * fill_price
                    remaining = 0.0
                    exit_reason = "Stop Loss"
                    exit_index = j
                    break

                if day_high >= target1:

                    booked = 50.0
                    remaining = 50.0

                    weighted_exit += 50.0 * target1
                    t1_hit = True

                    # Conservative handling of same-day ambiguity.
                    if day_low <= entry:

                        weighted_exit += 50.0 * entry
                        remaining = 0.0
                        exit_reason = "Target 1 + Breakeven"
                        exit_index = j
                        break

                    if day_high >= target2:

                        weighted_exit += 50.0 * target2
                        remaining = 0.0
                        exit_reason = "Target 1 + Target 2"
                        exit_index = j
                        break

                    continue

            else:

                # Remaining 50% protected at entry price.
                if day_low <= entry:

                    weighted_exit += remaining * entry
                    remaining = 0.0
                    exit_reason = "Partial + Breakeven"
                    exit_index = j
                    break

                if day_high >= target2:

                    weighted_exit += remaining * target2
                    remaining = 0.0
                    exit_reason = "Target 1 + Target 2"
                    exit_index = j
                    break

        if remaining > 0:

            final_price = float(
                df["Close"].iloc[last_index]
            )

            weighted_exit += remaining * final_price
            exit_index = last_index

            exit_reason = (
                "Partial + Time Exit"
                if t1_hit
                else "Time Exit"
            )

        average_exit = weighted_exit / 100.0

        gross_return = (
            (average_exit / entry) - 1
        ) * 100

        net_return = gross_return - cost

        trades.append({
            "Entry Date": str(
                df.index[entry_index].date()
            ),
            "Exit Date": str(
                df.index[exit_index].date()
            ),
            "Entry ₹": round(entry, 2),
            "Stock": symbol,
            "Stop ₹": round(stop, 2),
            "Target 1 ₹": round(target1, 2),
            "Target 2 ₹": round(target2, 2),
            "Exit ₹": round(average_exit, 2),
            "Target 1 Booked %": booked,
            "Exit Reason": exit_reason,
            "Return %": round(net_return, 2)
        })

        i = exit_index + 1

    return trades
def simulate_portfolio(
    trades,
    initial_capital=10000,
    risk_pct=1.0,
    max_positions=5,
    allocation_pct=20.0,
    trading_cost_pct=0.20
):
    if not trades:
        return [], {}

    candidates = pd.DataFrame(trades).copy()

    required = [
        "Stock", "Entry Date", "Exit Date",
        "Entry ₹", "Exit ₹", "Stop ₹",
        "Return %", "Exit Reason"
    ]

    if not all(col in candidates.columns for col in required):
        raise ValueError(
            "Portfolio backtest: required trade fields missing."
        )

    initial_capital = float(initial_capital)
    risk_pct = float(risk_pct)
    allocation_pct = float(allocation_pct)
    trading_cost_pct = float(trading_cost_pct)
    max_positions = int(max_positions)

    if initial_capital <= 0 or risk_pct <= 0:
        raise ValueError("Capital and risk must be positive.")

    if (
        max_positions < 1
        or not 0 < allocation_pct <= 100
        or trading_cost_pct < 0
    ):
        raise ValueError("Invalid portfolio settings.")

    candidates["Entry Date"] = pd.to_datetime(
        candidates["Entry Date"], errors="coerce"
    )

    candidates["Exit Date"] = pd.to_datetime(
        candidates["Exit Date"], errors="coerce"
    )

    for col in [
        "Entry ₹", "Exit ₹", "Stop ₹", "Return %"
    ]:
        candidates[col] = pd.to_numeric(
            candidates[col], errors="coerce"
        )

    candidates = candidates.dropna(
        subset=required[:-1]
    )

    candidates = candidates.sort_values(
        ["Entry Date", "Stock"]
    ).reset_index(drop=True)

    cash = initial_capital
    equity = initial_capital

    open_positions = []
    completed = []

    for _, trade in candidates.iterrows():

        entry_date = trade["Entry Date"]
        entry = float(trade["Entry ₹"])
        exit_price = float(trade["Exit ₹"])
        stop = float(trade["Stop ₹"])
        net_return = float(trade["Return %"])
        stock = str(trade["Stock"])

        if (
            entry <= 0
            or exit_price <= 0
            or stop <= 0
            or stop >= entry
            or trade["Exit Date"] < entry_date
            or net_return <= -100
        ):
            continue

        # Release positions exited before the new entry date.
        remaining = []

        for pos in open_positions:

            if pos["Exit Date"] < entry_date:

                cash += (
                    pos["Invested ₹"] + pos["P&L ₹"]
                )

                equity += pos["P&L ₹"]
                completed.append(pos)

            else:
                remaining.append(pos)

        open_positions = remaining

        # Avoid overlapping trades in the same stock.
        if any(
            pos["Stock"] == stock
            for pos in open_positions
        ):
            continue

        if len(open_positions) >= max_positions:
            continue

        risk_per_share = (
            entry - stop
            + entry * trading_cost_pct / 100
        )

        if risk_per_share <= 0:
            continue

        risk_amount = equity * risk_pct / 100
        allocation_limit = equity * allocation_pct / 100

        qty_risk = int(risk_amount / risk_per_share)
        qty_allocation = int(allocation_limit / entry)
        qty_cash = int(cash / entry)

        quantity = min(
            qty_risk,
            qty_allocation,
            qty_cash
        )

        if quantity <= 0:
            continue

        invested = quantity * entry
        pnl = invested * net_return / 100

        position = {
            "Stock": stock,
            "Entry Date": entry_date,
            "Exit Date": trade["Exit Date"],
            "Entry ₹": round(entry, 2),
            "Exit ₹": round(exit_price, 2),
            "Stop ₹": round(stop, 2),
            "Target 1 ₹": trade.get("Target 1 ₹", np.nan),
            "Target 2 ₹": trade.get("Target 2 ₹", np.nan),
            "Target 1 Booked %": trade.get(
                "Target 1 Booked %", 0
            ),
            "Quantity": quantity,
            "Invested ₹": round(invested, 2),
            "P&L ₹": round(pnl, 2),
            "Return %": round(net_return, 2),
            "Exit Reason": trade["Exit Reason"]
        }

        cash -= invested
        open_positions.append(position)

    # Close all remaining positions.
    for pos in sorted(
        open_positions,
        key=lambda x: (x["Exit Date"], x["Stock"])
    ):

        cash += (
            pos["Invested ₹"] + pos["P&L ₹"]
        )

        equity += pos["P&L ₹"]
        completed.append(pos)

    result = pd.DataFrame(completed)

    if result.empty:
        return [], {}

    result = result.sort_values(
        ["Exit Date", "Entry Date", "Stock"]
    ).reset_index(drop=True)

    # Calculate realized daily equity and drawdown.
    daily_pnl = result.groupby("Exit Date")["P&L ₹"].sum()

    daily_equity = initial_capital + daily_pnl.cumsum()

    equity_curve = pd.Series(
        [initial_capital] + daily_equity.tolist()
    )

    peak = equity_curve.cummax()

    drawdown = (
        equity_curve / peak - 1
    ) * 100

    wins = result.loc[
        result["P&L ₹"] > 0, "P&L ₹"
    ].sum()

    losses = abs(
        result.loc[
            result["P&L ₹"] < 0, "P&L ₹"
        ].sum()
    )

    final_equity = (
        initial_capital + result["P&L ₹"].sum()
    )

    metrics = {
        "Trades": len(result),

        "Win Rate %": round(
            (result["P&L ₹"] > 0).mean() * 100, 1
        ),

        "Profit Factor": round(
            wins / losses, 2
        ) if losses > 0 else "No Losses",

        "Max Drawdown %": round(
            drawdown.min(), 2
        ),

        "Total Return %": round(
            (final_equity / initial_capital - 1) * 100, 2
        ),

        "Final Equity ₹": round(final_equity, 2)
    }

    result["Entry Date"] = result["Entry Date"].dt.date
    result["Exit Date"] = result["Exit Date"].dt.date

    return result.to_dict("records"), metrics

initial_capital = st.number_input(
    "Starting Capital (₹)",
    min_value=1000,
    value=10000,
    step=1000
)

risk_pct = st.number_input(
    "Risk per Trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

max_positions = st.number_input(
    "Maximum Open Positions",
    min_value=1,
    max_value=20,
    value=5
)

allocation_pct = st.number_input(
    "Maximum Allocation per Stock (%)",
    min_value=5.0,
    max_value=100.0,
    value=20.0,
    step=5.0
    )
if st.button("SCAN STOCKS", type="primary"):

    if min_price >= max_price:
        st.error("Maximum price must exceed minimum price.")

    elif not symbols:
        st.error("Please enter NSE symbols.")

    else:

        results = []
        errors = []
        all_trades = []
        stock_reports = []

        progress = st.progress(0)
        message = st.empty()

        for i, symbol in enumerate(symbols):

            message.write(
                f"Scanning {symbol} ({i+1}/{len(symbols)})"
            )

            try:

                raw = download_stock(symbol)

                if raw is None or len(raw) < 30:
                    errors.append({
                        "Stock": symbol,
                        "Reason": "Insufficient data"
                    })

                else:

                    prepared = calculate_indicators(raw)

                    result = analyze_stock(
                        symbol,
                        raw
                    )

                    if result is not None:

                        if min_price <= result["Price ₹"] <= max_price:
                            results.append(result)

                    trades = run_backtest(
    prepared,
    trading_cost,
    holding_days,
    symbol.replace(".NS", "")
)

                    if trades:

                        for trade in trades:
                            trade["Stock"] = symbol.replace(
                                ".NS", ""
                            )

                        all_trades.extend(trades)

                        metrics = calculate_metrics(trades)

                        if metrics:
                            metrics["Stock"] = symbol.replace(
                                ".NS", ""
                            )

                            stock_reports.append(metrics)

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

        # ==========================================
        # TOP 10 SCANNER RESULT COLUMN ORDER
        # ==========================================

        df_result["Remark / Note"] = (
            f"SHORT TERM | Daily Setup | "
            f"Max Holding: {holding_days} Trading Days"
        )

        priority_columns = [
            "Stock",
            "Trend",
            "Price ₹",
            "Entry Ref ₹",
            "Stop Loss Ref ₹",
            "Target 1 ₹",
            "Target 2 ₹",
            "Strength",
            "Signal"
        ]

        remaining_columns = [
            col for col in df_result.columns
            if col not in priority_columns
            and col != "Remark / Note"
        ]

        df_result = df_result[
            [col for col in priority_columns
             if col in df_result.columns]
            + remaining_columns
            + ["Remark / Note"]
        ]

        st.subheader(
            f"Qualifying Stocks: {len(df_result)}"
        )

        if len(df_result) > 0:

            selected_stock = st.selectbox(
                "📊 Select Stock to View Candlestick Chart",
                df_result["Stock"].dropna().unique().tolist(),
                key="selected_chart_stock"
            )
           # ==========================================
        # TOP 10 SCANNER RESULTS
        # ==========================================

        if results:

            df_result = pd.DataFrame(results)

            df_result = df_result.sort_values(
                ["Score", "Volume Ratio"],
                ascending=False,
                na_position="last"
            ).head(max_stocks)

            df_result["Remark / Note"] = (
                f"SHORT TERM | Daily Setup | "
                f"Max Holding: {holding_days} Trading Days"
            )

            priority_columns = [
                "Stock",
                "Trend",
                "Price ₹",
                "Entry Ref ₹",
                "Stop Loss Ref ₹",
                "Target 1 ₹",
                "Target 2 ₹",
                "Strength",
                "Signal"
            ]

            remaining_columns = [
                col for col in df_result.columns
                if col not in priority_columns
                and col != "Remark / Note"
            ]

            df_result = df_result[
                [col for col in priority_columns
                 if col in df_result.columns]
                + remaining_columns
                + ["Remark / Note"]
            ]

            st.subheader(
                f"Qualifying Stocks: {len(df_result)}"
            )

            # Scanner Table
            st.dataframe(
                df_result,
                use_container_width=True,
                hide_index=True
            )

            # CSV Download
            st.download_button(
                "Download Scan CSV",
                df_result.to_csv(index=False).encode("utf-8"),
                file_name=f"SSA_{date.today()}.csv",
                mime="text/csv"
            )

            # Save Scan History
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

            # ==========================================
            # SELECT STOCK
            # ==========================================

            selected_stock = st.selectbox(
    "📊 Select Stock to View Candlestick Chart",
    df_result["Stock"].dropna().unique().tolist(),
    key="scanner_chart_stock_dropdown"
)

            # IMPORTANT: Add .NS because scanner stores
            # stock names without the Yahoo Finance suffix.
            chart_df = download_stock(
                selected_stock + ".NS"
            )

            # ==========================================
            # ADVANCED CANDLESTICK CHART
            # ==========================================

            st.subheader(
                f"📊 Technical Chart: {selected_stock}"
            )

            if chart_df is not None and not chart_df.empty:

                chart_df["EMA9"] = (
                    chart_df["Close"].ewm(
                        span=9,
                        adjust=False
                    ).mean()
                )

                chart_df["EMA20"] = (
                    chart_df["Close"].ewm(
                        span=20,
                        adjust=False
                    ).mean()
                )

                chart_df["EMA50"] = (
                    chart_df["Close"].ewm(
                        span=50,
                        adjust=False
                    ).mean()
                )

                # EMA Crossover Signals
                chart_df["BUY"] = (
                    (chart_df["EMA9"] > chart_df["EMA20"]) &
                    (chart_df["EMA9"].shift(1) <=
                     chart_df["EMA20"].shift(1))
                )

                chart_df["SELL"] = (
                    (chart_df["EMA9"] < chart_df["EMA20"]) &
                    (chart_df["EMA9"].shift(1) >=
                     chart_df["EMA20"].shift(1))
                )

                selected_row = df_result[
                    df_result["Stock"] == selected_stock
                ].iloc[0]

                fig = go.Figure()

                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=chart_df.index,
                        open=chart_df["Open"],
                        high=chart_df["High"],
                        low=chart_df["Low"],
                        close=chart_df["Close"],
                        name="Price"
                    )
                )

                # EMA Lines
                for ema, label in [
                    ("EMA9", "EMA 9"),
                    ("EMA20", "EMA 20"),
                    ("EMA50", "EMA 50")
                ]:

                    fig.add_trace(
                        go.Scatter(
                            x=chart_df.index,
                            y=chart_df[ema],
                            mode="lines",
                            name=label
                        )
                    )

                # BUY Markers
                buy_points = chart_df[
                    chart_df["BUY"]
                ]

                fig.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points["Low"] * 0.995,
                        mode="markers+text",
                        name="BUY",
                        text=["BUY"] * len(buy_points),
                        textposition="bottom center"
                    )
                )

                # SELL Markers
                sell_points = chart_df[
                    chart_df["SELL"]
                ]

                fig.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points["High"] * 1.005,
                        mode="markers+text",
                        name="SELL",
                        text=["SELL"] * len(sell_points),
                        textposition="top center"
                    )
                )

                # Entry, Stop Loss and Targets
                levels = [
                    ("Entry Ref ₹", "ENTRY"),
                    ("Stop Loss Ref ₹", "STOP LOSS"),
                    ("Target 1 ₹", "TARGET 1"),
                    ("Target 2 ₹", "TARGET 2")
                ]

                for column, label in levels:

                    value = pd.to_numeric(
                        selected_row.get(column),
                        errors="coerce"
                    )

                    if pd.notna(value) and value > 0:

                        fig.add_hline(
                            y=value,
                            line_dash="dash",
                            annotation_text=(
                                f"{label}: ₹{value:.2f}"
                            )
                        )

                # Resistance
                resistance = (
                    chart_df["High"].tail(20).max()
                )

                fig.add_hline(
                    y=resistance,
                    line_dash="dot",
                    annotation_text=(
                        f"RESISTANCE: ₹{resistance:.2f}"
                    )
                )

                # Support
                support = (
                    chart_df["Low"].tail(20).min()
                )

                fig.add_hline(
                    y=support,
                    line_dash="dot",
                    annotation_text=(
                        f"SUPPORT: ₹{support:.2f}"
                    )
                )

                fig.update_layout(
                    title=(
                        f"{selected_stock} - Daily Candlestick Chart"
                    ),
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    xaxis_rangeslider_visible=False,
                    height=700,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Chart data is currently unavailable."
                )

        else:

            st.warning(
                "No stocks matched your price filter."
            )

        # ==========================================
        # ADVANCED BACKTESTING REPORT
        # ==========================================

        st.divider()

        st.header(
            "📈 Advanced Historical Backtesting"
        )

        if all_trades:

            portfolio_trades, overall = simulate_portfolio(
                all_trades,
                initial_capital=initial_capital,
                risk_pct=risk_pct,
                max_positions=max_positions,
                allocation_pct=allocation_pct,
                trading_cost_pct=trading_cost
            )

            st.subheader("Overall Strategy Performance")  


            st.subheader("Overall Strategy Performance")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Trades", overall["Trades"])
            c2.metric("Win Rate", f'{overall["Win Rate %"]}%')
            c3.metric("Profit Factor", overall["Profit Factor"])
            c4.metric("Max Drawdown", f'{overall["Max Drawdown %"]}%')

            c5, c6, c7 = st.columns(3)

            c5.metric("Starting Capital", f'₹{initial_capital:,.0f}')
            c6.metric("Final Capital", f'₹{overall["Final Equity ₹"]:,.2f}')
            c7.metric(
                "Net Profit / Loss",
                f'₹{overall["Final Equity ₹"] - initial_capital:,.2f}'
            )

            st.metric("Portfolio Return", f'{overall["Total Return %"]}%')

            st.subheader("Stock-wise Backtest Report")

            report_df = pd.DataFrame(stock_reports)
            if not report_df.empty:

                report_df = report_df.sort_values(
                    "Win Rate %",
                    ascending=False
                )

                st.dataframe(
                    report_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Backtest Report",
                    report_df.to_csv(index=False).encode("utf-8"),
                    file_name="SSA_Backtest_Report.csv",
                    mime="text/csv"
                )

            trade_df = pd.DataFrame(portfolio_trades)

            st.subheader("Historical Trade Records")

            st.dataframe(
                trade_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download All Historical Trades",
                trade_df.to_csv(index=False).encode("utf-8"),
                file_name="SSA_Historical_Trades.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "Not enough historical data or no qualifying "
                "BUY WATCH trades found for backtesting."
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
            "Backtest rules: BUY WATCH score >= 5; "
            "next-day Open entry; 1.5 ATR stop; "
            "2 ATR target; maximum holding period as selected. "
            "If stop and target are touched on the same candle, "
            "stop is assumed first. One open trade per stock. "
            "Costs are estimated, not actual broker charges."
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
    st.write("No scan history in this session.")

st.caption(
    "SSA v0.4 | Advanced Backtesting | Manual Trading Only"
)
