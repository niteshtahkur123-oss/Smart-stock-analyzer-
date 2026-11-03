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


def run_backtest(df, cost, max_days, symbol=""):

    trades = []

    if len(df) < 220:
        return trades

    # Start only after enough EMA200 history.
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

        # Signal after today's close.
        # Entry on next trading day's open.
        entry_index = i + 1

        entry = float(df["Open"].iloc[entry_index])

        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue

        stop = entry - 1.5 * atr
        target = entry + 2 * atr

        last_index = min(
            entry_index + max_days - 1,
            len(df) - 1
        )

        exit_price = None
        exit_reason = "Time Exit"
        exit_index = last_index

        for j in range(entry_index, last_index + 1):

            day_low = float(df["Low"].iloc[j])
            day_high = float(df["High"].iloc[j])

            # Conservative rule if both levels are touched.
            if day_low <= stop:
                exit_price = stop
                exit_reason = "Stop Loss"
                exit_index = j
                break

            if day_high >= target:
                exit_price = target
                exit_reason = "Target"
                exit_index = j
                break

        if exit_price is None:
            exit_price = float(df["Close"].iloc[last_index])

        gross_return = (
            (exit_price / entry) - 1
        ) * 100

        net_return = gross_return - cost

        trades.append({
            "Entry Date": str(
                df.index[entry_index].date()
            ),
            "Exit Date": str(df.index[exit_index].date()),
"Entry ₹": round(entry, 2),
            "Stock": symbol,
"Stop ₹": round(stop, 2),
            "Exit ₹": round(exit_price, 2),
            "Exit Reason": exit_reason,
            "Return %": round(net_return, 2)
        })

        # One trade at a time per stock.
        i = exit_index + 1

    return trades


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
def simulate_portfolio(
    trades,
    initial_capital=10000,
    risk_pct=1.0,
    max_positions=5,
    allocation_pct=20.0
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

    candidates["Entry Date"] = pd.to_datetime(
        candidates["Entry Date"]
    )
    candidates["Exit Date"] = pd.to_datetime(
        candidates["Exit Date"]
    )

    candidates = candidates.sort_values(
        ["Entry Date", "Stock"]
    ).reset_index(drop=True)

    cash = float(initial_capital)
    equity = float(initial_capital)

    open_positions = []
    completed = []

    for _, trade in candidates.iterrows():

        entry_date = trade["Entry Date"]

        # पहले पुराने positions close करें.
        remaining = []

        for pos in open_positions:

            if pd.Timestamp(pos["Exit Date"]).date() <= pd.Timestamp(entry_date).date():

                cash += pos["Invested ₹"] + pos["P&L ₹"]
                equity += pos["P&L ₹"]
                completed.append(pos)

            else:
                remaining.append(pos)

        open_positions = remaining

        if len(open_positions) >= max_positions:
            continue

        entry = float(trade["Entry ₹"])
        exit_price = float(trade["Exit ₹"])
        stop = float(trade["Stop ₹"])
        net_return = float(trade["Return %"])

        if entry <= 0 or stop <= 0 or stop >= entry:
            continue

        risk_per_share = entry - stop

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
            "Stock": trade["Stock"],
            "Entry Date": entry_date.date(),
            "Exit Date": trade["Exit Date"].date(),
            "Entry ₹": round(entry, 2),
"Exit ₹": round(exit_price, 2),
"Stop ₹": round(stop, 2),
"Quantity": quantity,
"Invested ₹": round(invested, 2),
"P&L ₹": round(pnl, 2),
            "Return %": round(net_return, 2),
            "Exit Reason": trade["Exit Reason"]
        }

        cash -= invested
        open_positions.append(position)


    for pos in sorted(
        open_positions,
        key=lambda x: x["Exit Date"]
    ):
        cash += pos["Invested ₹"] + pos["P&L ₹"]
        equity += pos["P&L ₹"]
        completed.append(pos)

    result = pd.DataFrame(completed)

    if result.empty:
        return [], {}

    result = result.sort_values(
        ["Exit Date", "Entry Date", "Stock"]
    ).reset_index(drop=True)

    result["Equity ₹"] = (
        initial_capital + result["P&L ₹"].cumsum()
    )

    peak = result["Equity ₹"].cummax().clip(
        lower=initial_capital
    )

    drawdown = (
        result["Equity ₹"] / peak - 1
    ) * 100

    wins = result.loc[
        result["P&L ₹"] > 0, "P&L ₹"
    ].sum()

    losses = abs(
        result.loc[
            result["P&L ₹"] <= 0, "P&L ₹"
        ].sum()
    )

    final_equity = initial_capital + result["P&L ₹"].sum()

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

    return result.to_dict("records"), metrics
st.subheader("Portfolio Backtest Settings")

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
                df_result.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"SSA_{date.today()}.csv",
                mime="text/csv"
            )

            history = df_result.copy()

            history.insert(
                0,
                "Scan Time",
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            st.session_state.scan_history = pd.concat(
                [
                    st.session_state.scan_history,
                    history
                ],
                ignore_index=True
            )

        else:
            st.warning(
                "No stocks matched your price filter."
            )

                        # Advanced backtesting report
        st.divider()
        st.header("📈 Advanced Historical Backtesting")
        
        if all_trades:

            portfolio_trades, overall = simulate_portfolio(
                all_trades,
                initial_capital=initial_capital,
                risk_pct=risk_pct,
                max_positions=max_positions,
                allocation_pct=allocation_pct
            )

            st.subheader("Overall Strategy Performance")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Trades", overall["Trades"])
            c2.metric("Win Rate", f'{overall["Win Rate %"]}%')
            c3.metric("Profit Factor", overall["Profit Factor"])
            c4.metric(
    "Max Drawdown",
    f'{overall.get("Max Drawdown %", overall.get("Max Drawdown", 0))}%'
            )

            c5, c6, c7 = st.columns(3)

            c5.metric(
                "Starting Capital",
                f'₹{initial_capital:,.0f}'
            )

            c6.metric(
                "Final Capital",
                f'₹{overall["Final Equity ₹"]:,.2f}'
            )

            c7.metric(
                "Net Profit / Loss",
                f'₹{overall["Final Equity ₹"] - initial_capital:,.2f}'
            )

            st.metric(
                "Portfolio Return",
                f'{overall["Total Return %"]}%'
            )

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
