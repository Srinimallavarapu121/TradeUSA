"""
ETH-USD short-timeframe chart with EMA, RSI, and VWAP overlays.

Educational/analysis tool only — does NOT predict price direction.
Short-timeframe (15-minute) crypto price movement is highly noisy;
no indicator combination here should be treated as a trading signal.

Usage:
    python eth_chart.py                        # all six timeframes (1m–1d)
    python eth_chart.py --interval 5m           # single timeframe
    python eth_chart.py --save eth_{interval}.png  # save images
"""

import argparse
import sys

import numpy as np
import mplfinance as mpf
import pandas as pd
import yfinance as yf


def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    # yfinance doesn't support 4h natively; fetch 1h and resample
    fetch_interval = "1h" if interval == "4h" else interval
    df = yf.download(ticker, interval=fetch_interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(
            f"No data returned for {ticker} (interval={interval}, period={period}). "
            "yfinance limits intraday history (e.g. 15m data is capped at ~60 days)."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if interval == "4h":
        df = df.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # Buy/sell signals: high volume + positive candle = BUY, low volume + negative candle = SELL
    price_change = df["Close"] - df["Open"]
    vol_mean = df["Volume"].rolling(20).mean()
    high_vol = df["Volume"] > vol_mean
    low_vol = df["Volume"] < vol_mean

    df["BUY"] = np.where(high_vol & (price_change > 0), df["Low"] * 0.998, np.nan)
    df["SELL"] = np.where(low_vol & (price_change < 0), df["High"] * 1.002, np.nan)

    return df


# yfinance caps intraday history differently per interval
DEFAULT_PERIODS = {"1m": "1d", "5m": "2d", "15m": "5d", "1h": "30d", "4h": "60d", "1d": "6mo"}
ALL_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]


def plot_chart(df: pd.DataFrame, ticker: str, interval: str, save_path: str | None) -> None:
    overlays = [
        mpf.make_addplot(df["EMA9"], color="orange", width=1.0),
        mpf.make_addplot(df["EMA21"], color="blue", width=1.0),
        mpf.make_addplot(df["VWAP"], color="purple", width=1.0, linestyle="--"),
        mpf.make_addplot(df["BUY"], type="scatter", marker="^", markersize=60, color="green"),
        mpf.make_addplot(df["SELL"], type="scatter", marker="v", markersize=60, color="red"),
        mpf.make_addplot(df["RSI14"], panel=2, color="green", ylabel="RSI"),
    ]

    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        addplot=overlays,
        volume=True,
        panel_ratios=(6, 2, 2),
        title=f"{ticker} ({interval}) — BUY(▲green) / SELL(▼red)",
        savefig=save_path if save_path else None,
        show_nontrading=False,
        warn_too_much_data=5000,
    )


def run_single(ticker: str, interval: str, period: str, save_path: str | None) -> None:
    df = fetch_data(ticker, interval, period)
    df = add_indicators(df)
    plot_chart(df, ticker, interval, save_path)
    if save_path:
        print(f"  [{interval}] saved → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="ETH-USD", help="Yahoo Finance ticker (default: ETH-USD)")
    parser.add_argument("--interval", default=None,
                        help="Single interval (1m, 5m, 15m). Omit to generate all three.")
    parser.add_argument("--period", default=None,
                        help="Lookback period (e.g. 5d). Auto-selected per interval if omitted.")
    parser.add_argument("--save", default=None,
                        help="Save path. Use {interval} placeholder for multi-chart, e.g. eth_{interval}.png")
    args = parser.parse_args()

    intervals = [args.interval] if args.interval else ALL_INTERVALS

    try:
        for ivl in intervals:
            period = args.period or DEFAULT_PERIODS.get(ivl, "5d")
            save = args.save.replace("{interval}", ivl) if args.save else None
            run_single(args.ticker, ivl, period, save)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
