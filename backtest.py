"""
Causal backtest for the dashboard's two prediction engines:
  1. Directional Bias (15 min)  - predicts UP/DOWN 15 minutes ahead
  2. Target-crossing probability (breakoutProb) - predicts P(touch X% move within 15 min)

Replicates the exact JS math from stock_details.html and walks forward through
real historical 1m ETHUSDT candles. At each simulated "now" (T), only bars up
to T are used (no lookahead) -- matched historical samples must have their own
15-min outcome already resolved by T (j + 15 <= T) to avoid cheating.
"""
import json
import math
import time
import urllib.request

SYMBOL = "ETHUSDT"
N_TARGET = 10000         # ~7 days of 1m candles
HORIZON = 15
DECAY = 0.006
TARGET_PCTS = [0.05, 0.10, 0.20, 0.30]
OUT_OF_SAMPLE_OFFSET_DAYS = 7   # fetch the PRIOR week, non-overlapping with the window CALIBRATION_CURVE was derived from
WALKFORWARD_N_TARGET = 3000     # ~2 days/window; keeps 10-window validation bounded and reproducible

# Exact copy of CALIBRATION_CURVE / calibrateProb() from stock_details.html
CALIBRATION_CURVE = [(0.0, 0.0), (3.7, 7.6), (15.2, 20.5), (24.8, 26.3), (34.9, 34.6),
                      (45.1, 44.0), (55.0, 55.3), (65.1, 63.3), (75.0, 72.9),
                      (83.7, 74.7), (92.1, 77.1), (100.0, 77.1)]


def calibrate_prob(raw):
    pts = CALIBRATION_CURVE
    if raw <= pts[0][0]:
        return pts[0][1]
    for i in range(1, len(pts)):
        if raw <= pts[i][0]:
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            t = (raw - x0) / (x1 - x0) if x1 > x0 else 0
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def fetch_klines(total, end_time=None):
    """Paginate data-api.binance.vision klines backwards from end_time (or now)."""
    out = []
    while len(out) < total:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={SYMBOL}&interval=1m&limit=1000"
        if end_time:
            url += f"&endTime={end_time}"
        with urllib.request.urlopen(url, timeout=15) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        out = batch + out
        end_time = batch[0][0] - 1
        time.sleep(0.15)
    return out[-total:]


def build_series(raw):
    o = [float(c[1]) for c in raw]
    h = [float(c[2]) for c in raw]
    l = [float(c[3]) for c in raw]
    c = [float(c[4]) for c in raw]
    v = [float(c[5]) for c in raw]
    return o, h, l, c, v


def rolling_atr(h, l, c, n):
    atr = [0.0] * n
    for i in range(19, n):
        s = 0.0
        for j in range(i - 19, i + 1):
            s += (h[j] - l[j]) / c[j] * 100
        atr[i] = s / 20
    return atr


def rolling_mom(o, c, n):
    mom = ["flat"] * n
    for i in range(20, n):
        ret = (c[i] - o[i - 20]) / o[i - 20] * 100
        mom[i] = "up" if ret > 0.02 else "down" if ret < -0.02 else "flat"
    return mom


def live_momentum_label(o, c, index, window=20):
    """Match breakoutProb's current trend label over its last `window` candles."""
    first = max(0, index - window + 1)
    change_pct = (c[index] - o[first]) / o[first] * 100
    return "up" if change_pct > 0.02 else "down" if change_pct < -0.02 else "flat"


def rolling_ema(series, n, period):
    """Standard causal EMA (each value depends only on itself and prior values)."""
    ema = [0.0] * n
    k = 2 / (period + 1)
    ema[0] = series[0]
    for i in range(1, n):
        ema[i] = series[i] * k + ema[i - 1] * (1 - k)
    return ema


def rolling_rsi(c, n, period=14):
    """Wilder's RSI, causal."""
    rsi = [50.0] * n
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = c[i] - c[i - 1]
        gains[i] = max(diff, 0.0)
        losses[i] = max(-diff, 0.0)
    avg_gain = avg_loss = 0.0
    for i in range(1, n):
        if i < period:
            continue
        if i == period:
            avg_gain = sum(gains[1:period + 1]) / period
            avg_loss = sum(losses[1:period + 1]) / period
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi


def rolling_macd(c, n, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram -- all causal EMAs."""
    ema_fast = rolling_ema(c, n, fast)
    ema_slow = rolling_ema(c, n, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(n)]
    signal_line = rolling_ema(macd_line, n, signal)
    hist = [macd_line[i] - signal_line[i] for i in range(n)]
    return macd_line, signal_line, hist


def combo_signal_backtest(c, n, ema9, ema21, rsi14, hist, horizon):
    """EMA9/21 trend + RSI14 momentum + MACD histogram, all must agree to fire a call.
    This is the classic 'triple confirmation' trend-following setup (Elder's Triple Screen
    style). No call is made when the three indicators disagree -- an honest no-signal,
    same convention as the existing FLAT/Neutral no-call buckets in this file."""
    warmup = 40  # let EMA21 / MACD(26,9) / RSI14 stabilize
    records = []  # (signal or None, actual_up)
    for T in range(warmup, n - horizon):
        ema_trend = "up" if ema9[T] > ema21[T] else "down"
        rsi_state = "bull" if rsi14[T] > 50 else "bear"
        macd_state = "bull" if hist[T] > 0 else "bear"

        if ema_trend == "up" and rsi_state == "bull" and macd_state == "bull":
            signal = "UP"
        elif ema_trend == "down" and rsi_state == "bear" and macd_state == "bear":
            signal = "DOWN"
        else:
            signal = None

        actual_up = 1 if c[T + horizon] > c[T] else 0
        records.append((signal, actual_up))
    return records


def report_combo_backtest(records, horizon, title="COMBO INDICATOR (EMA9/21 + RSI14 + MACD 12/26/9)"):
    print("=" * 70)
    print(f"{title} BACKTEST -- {horizon} MIN HORIZON")
    print("=" * 70)
    calls = [r for r in records if r[0] is not None]
    coverage = len(calls) / len(records) * 100 if records else 0
    correct = sum(1 for s, a in calls if (s == "UP") == (a == 1))
    acc = correct / len(calls) * 100 if calls else 0
    print(f"Total points: {len(records)}   Calls made: {coverage:.1f}%  (n={len(calls)})")
    print(f"Directional accuracy on calls: {acc:.1f}%  ({correct}/{len(calls)})")
    up_calls = [r for r in calls if r[0] == "UP"]
    down_calls = [r for r in calls if r[0] == "DOWN"]
    if up_calls:
        up_acc = sum(1 for s, a in up_calls if a == 1) / len(up_calls) * 100
        print(f"  UP calls:   {up_acc:5.1f}%  (n={len(up_calls)})")
    if down_calls:
        down_acc = sum(1 for s, a in down_calls if a == 0) / len(down_calls) * 100
        print(f"  DOWN calls: {down_acc:5.1f}%  (n={len(down_calls)})")
    print()


def rolling_bollinger(c, n, period=20, mult=2.0):
    """Bollinger Bands, causal (window looks only at bars up to and including i)."""
    mid = [0.0] * n
    upper = [0.0] * n
    lower = [0.0] * n
    for i in range(period - 1, n):
        window = c[i - period + 1:i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        mid[i] = m
        upper[i] = m + mult * sd
        lower[i] = m - mult * sd
    return mid, upper, lower


def bollinger_reversion_backtest(c, n, upper, lower, horizon, warmup=25):
    """Classic mean-reversion: fade closes outside the bands (expect snap-back to mid)."""
    records = []
    for T in range(warmup, n - horizon):
        if upper[T] == 0:
            continue
        if c[T] > upper[T]:
            signal = "DOWN"
        elif c[T] < lower[T]:
            signal = "UP"
        else:
            signal = None
        actual_up = 1 if c[T + horizon] > c[T] else 0
        records.append((signal, actual_up))
    return records


def rsi_extreme_backtest(c, n, rsi14, horizon, warmup=25, lo_th=30, hi_th=70):
    """Classic mean-reversion: fade RSI overbought/oversold extremes."""
    records = []
    for T in range(warmup, n - horizon):
        if rsi14[T] < lo_th:
            signal = "UP"
        elif rsi14[T] > hi_th:
            signal = "DOWN"
        else:
            signal = None
        actual_up = 1 if c[T + horizon] > c[T] else 0
        records.append((signal, actual_up))
    return records


def combo_main():
    """Fast standalone check: does a well-known EMA/RSI/MACD confirmation strategy
    beat the already-proven-broken momentum-only direction call? Run with:
        python backtest.py combo
    """
    end_time = int(time.time() * 1000) - OUT_OF_SAMPLE_OFFSET_DAYS * 24 * 3600 * 1000
    print(f"Fetching ~{N_TARGET} 1m {SYMBOL} candles ending {OUT_OF_SAMPLE_OFFSET_DAYS} days ago "
          f"(out-of-sample window) ...")
    raw = fetch_klines(N_TARGET, end_time=end_time)
    n = len(raw)
    print(f"Got {n} candles  ({n/60/24:.2f} days)  "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[0][0]/1000))} UTC -> "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[-1][0]/1000))} UTC\n")

    o, h, l, c, v = build_series(raw)
    ema9 = rolling_ema(c, n, 9)
    ema21 = rolling_ema(c, n, 21)
    rsi14 = rolling_rsi(c, n, 14)
    _, _, hist = rolling_macd(c, n, 12, 26, 9)
    _, bb_upper, bb_lower = rolling_bollinger(c, n, 20, 2.0)

    for horizon in (5, 15):
        records = combo_signal_backtest(c, n, ema9, ema21, rsi14, hist, horizon)
        report_combo_backtest(records, horizon)

    print("\n" + "#" * 70)
    print("# MEAN-REVERSION STRATEGIES (opposite hypothesis to trend-following)")
    print("#" * 70 + "\n")
    for horizon in (5, 15):
        records = bollinger_reversion_backtest(c, n, bb_upper, bb_lower, horizon)
        report_combo_backtest(records, horizon, title="BOLLINGER BANDS (20,2) REVERSION")

    for horizon in (5, 15):
        records = rsi_extreme_backtest(c, n, rsi14, horizon)
        report_combo_backtest(records, horizon, title="RSI(14) OVERBOUGHT/OVERSOLD REVERSION")


def rsi_check_main():
    """Out-of-time robustness check for the RSI(14) reversion finding: re-test on a
    DIFFERENT, non-overlapping week (14-21 days back) and with a stricter 20/80
    threshold, to make sure the ~55-58% edge from the first window isn't a fluke.
    Run with: python backtest.py rsicheck
    """
    for offset_days in (7, 14):
        end_time = int(time.time() * 1000) - offset_days * 24 * 3600 * 1000
        print(f"\nFetching ~{N_TARGET} 1m {SYMBOL} candles ending {offset_days} days ago ...")
        raw = fetch_klines(N_TARGET, end_time=end_time)
        n = len(raw)
        print(f"Got {n} candles  ({n/60/24:.2f} days)  "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[0][0]/1000))} UTC -> "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[-1][0]/1000))} UTC")
        o, h, l, c, v = build_series(raw)
        rsi14 = rolling_rsi(c, n, 14)

        for lo_th, hi_th in ((30, 70), (20, 80)):
            for horizon in (5, 15):
                records = rsi_extreme_backtest(c, n, rsi14, horizon, lo_th=lo_th, hi_th=hi_th)
                report_combo_backtest(records, horizon,
                                       title=f"[offset={offset_days}d] RSI(14) {lo_th}/{hi_th} REVERSION")


def wilson_interval(correct, total, z=1.96):
    """Wilson 95% interval for a binomial accuracy proportion."""
    if total == 0:
        return [None, None]
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [(centre - margin) * 100, (centre + margin) * 100]


def max_losing_streak(values):
    longest = current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def strategy_metrics(records, cost_pct=0.30):
    """Calculate direction, calibration, and simple close-to-close net-return metrics.

    records are (signal, actual_up, signed_return_pct), where signed_return_pct is
    the close-to-close return in the predicted direction before costs. A trade pays
    the configured round-trip fee+slippage only when a signal exists.
    """
    calls = [record for record in records if record[0] is not None]
    total = len(calls)
    correct = sum(1 for signal, actual_up, _ in calls if (signal == "UP") == (actual_up == 1))
    net_returns = []
    for signal, _, signed_return_pct in calls:
        direction_return = signed_return_pct if signal == "UP" else -signed_return_pct
        net_returns.append(direction_return - cost_pct)

    gains = sum(value for value in net_returns if value > 0)
    losses = abs(sum(value for value in net_returns if value < 0))
    equity = peak = drawdown = 0.0
    for value in net_returns:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)

    up_calls = [record for record in calls if record[0] == "UP"]
    down_calls = [record for record in calls if record[0] == "DOWN"]

    def call_accuracy(subset, expected_up):
        if not subset:
            return None
        return sum(1 for _, actual_up, _ in subset if actual_up == expected_up) / len(subset) * 100

    return {
        "total_points": len(records),
        "signals": total,
        "coverage_pct": total / len(records) * 100 if records else 0.0,
        "correct": correct,
        "accuracy_pct": correct / total * 100 if total else None,
        "accuracy_ci95_pct": wilson_interval(correct, total),
        "up_precision_pct": call_accuracy(up_calls, 1),
        "down_precision_pct": call_accuracy(down_calls, 0),
        "up_signals": len(up_calls),
        "down_signals": len(down_calls),
        "net_expectancy_pct": sum(net_returns) / total if total else None,
        "profit_factor": gains / losses if losses else None,
        "max_drawdown_pct": drawdown,
        "max_losing_streak": max_losing_streak(net_returns),
        "cost_pct_round_trip": cost_pct,
    }


def baseline_records(c, horizon, signal):
    records = []
    for index in range(40, len(c) - horizon):
        actual_up = 1 if c[index + horizon] > c[index] else 0
        signed_return = (c[index + horizon] - c[index]) / c[index] * 100
        records.append((signal, actual_up, signed_return))
    return records


def build_walkforward_records(o, h, l, c, v, horizon=15):
    """Build causal records for each strategy on one historical window."""
    n = len(c)
    rsi = rolling_rsi(c, n, 14)
    atr = rolling_atr(h, l, c, n)
    valid_atr = [value for value in atr if value > 0]
    average_atr = sum(valid_atr) / len(valid_atr) if valid_atr else 0.0
    ema9 = rolling_ema(c, n, 9)
    ema21 = rolling_ema(c, n, 21)
    _, _, macd_hist = rolling_macd(c, n, 12, 26, 9)
    strategy_records = {
        "always_up": [], "always_down": [], "random_50": [],
        "rsi": [], "rsi_normal": [], "ema_macd": []
    }
    regime_records = {"Quiet": [], "Normal": [], "Hot": []}

    for index in range(40, n - horizon):
        actual_up = 1 if c[index + horizon] > c[index] else 0
        signed_return = (c[index + horizon] - c[index]) / c[index] * 100
        ratio = atr[index] / average_atr if average_atr and atr[index] else 1.0
        regime = "Hot" if ratio >= 1.5 else "Normal" if ratio >= 0.8 else "Quiet"
        raw_rsi = "UP" if rsi[index] < 30 else "DOWN" if rsi[index] > 70 else None
        trend_signal = None
        if ema9[index] > ema21[index] and macd_hist[index] > 0:
            trend_signal = "UP"
        elif ema9[index] < ema21[index] and macd_hist[index] < 0:
            trend_signal = "DOWN"
        random_signal = "UP" if ((index * 1103515245 + 12345) & 0x7FFFFFFF) % 2 else "DOWN"

        strategy_records["always_up"].append(("UP", actual_up, signed_return))
        strategy_records["always_down"].append(("DOWN", actual_up, signed_return))
        strategy_records["random_50"].append((random_signal, actual_up, signed_return))
        strategy_records["rsi"].append((raw_rsi, actual_up, signed_return))
        strategy_records["rsi_normal"].append((raw_rsi if regime == "Normal" else None, actual_up, signed_return))
        strategy_records["ema_macd"].append((trend_signal, actual_up, signed_return))
        regime_records[regime].append((raw_rsi, actual_up, signed_return))

    return strategy_records, regime_records


def walkforward_main():
    """Validate preserved baselines over 10 non-overlapping real-data windows.

    Windows are seven days of 1m candles, ending 7, 14, ..., 70 days ago. The
    thresholds are fixed before the run; no window is selected because it wins.
    Run with: python backtest.py walkforward
    """
    offsets = list(range(7, 71, 7))
    window_results = []
    aggregate_records = {name: [] for name in (
        "always_up", "always_down", "random_50", "rsi", "rsi_normal", "ema_macd"
    )}

    for offset_days in offsets:
        end_time = int(time.time() * 1000) - offset_days * 24 * 3600 * 1000
        print(f"Fetching walk-forward window ending {offset_days} days ago ...")
        raw = fetch_klines(WALKFORWARD_N_TARGET, end_time=end_time)
        if len(raw) < 1000:
            print(f"  skipped: only {len(raw)} candles")
            continue
        o, h, l, c, v = build_series(raw)
        records, regimes = build_walkforward_records(o, h, l, c, v, HORIZON)
        result = {"offset_days": offset_days, "candles": len(raw), "strategies": {}}
        for name, strategy in records.items():
            result["strategies"][name] = strategy_metrics(strategy)
            aggregate_records[name].extend(strategy)
        result["regimes"] = {name: strategy_metrics(values) for name, values in regimes.items()}
        window_results.append(result)
        print(f"  RSI normal: {result['strategies']['rsi_normal']['accuracy_pct']:.1f}% "
              f"({result['strategies']['rsi_normal']['signals']} signals), "
              f"RSI ungated: {result['strategies']['rsi']['accuracy_pct']:.1f}%")

    aggregate = {name: strategy_metrics(records) for name, records in aggregate_records.items()}
    gated = aggregate["rsi_normal"]
    ungated = aggregate["rsi"]
    gated_windows = [
        result["strategies"]["rsi_normal"] for result in window_results
        if result["strategies"]["rsi_normal"]["signals"] >= 50
    ]
    robust_accuracy = (
        gated["accuracy_pct"] > ungated["accuracy_pct"]
        and gated["accuracy_ci95_pct"][0] > 50
        and gated["net_expectancy_pct"] > 0
        and sum(metric["accuracy_pct"] > 50 for metric in gated_windows) >= max(1, len(gated_windows) - 1)
    )
    report = {
        "symbol": SYMBOL,
        "horizon_minutes": HORIZON,
        "window_days": WALKFORWARD_N_TARGET // (60 * 24),
        "offset_days": offsets,
        "cost_assumption": "0.10% taker fee + 0.05% slippage per side = 0.30% round trip",
        "lookahead_policy": "Indicators use data through T; outcome uses close at T+15; thresholds fixed before run.",
        "windows": window_results,
        "aggregate": aggregate,
        "promotion_gate": {
            "minimum_window_signals": 50,
            "required_positive_net_expectancy_after_costs": True,
            "required_accuracy_ci95_lower_bound_above_50_pct": True,
            "required_most_windows_above_50_pct": True,
            "passed": robust_accuracy,
        },
        "decision": "PROMOTE rsi_normal" if robust_accuracy else "DO NOT PROMOTE: candidate fails robustness and/or cost-adjusted expectancy gate.",
    }
    with open("walkforward_results.json", "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY (10 fixed, non-overlapping windows)")
    print("=" * 70)
    for name, metrics in aggregate.items():
        print(f"{name:12s}: accuracy={metrics['accuracy_pct']!s:>6}%, "
              f"CI95={metrics['accuracy_ci95_pct']}, signals={metrics['signals']}, "
              f"coverage={metrics['coverage_pct']:.1f}%, "
              f"net expectancy={metrics['net_expectancy_pct']!s}%")
    print("\nSaved machine-readable output to walkforward_results.json")


def rolling_horizon_extremes(h, l, c, n, max_horizon):
    """Precompute favorable wick excursions for every supported horizon."""
    up = [[0.0] * n for _ in range(max_horizon + 1)]
    down = [[0.0] * n for _ in range(max_horizon + 1)]
    for index in range(n - 1):
        highest = lowest = c[index]
        for horizon in range(1, min(max_horizon, n - index - 1) + 1):
            highest = max(highest, h[index + horizon])
            lowest = min(lowest, l[index + horizon])
            up[horizon][index] = (highest - c[index]) / c[index] * 100
            down[horizon][index] = (c[index] - lowest) / c[index] * 100
    return up, down


def target_touch_records(o, h, l, c, target_pcts=TARGET_PCTS, max_horizon=15,
                         lookback=500, sample_step=5):
    """Replay breakoutProb causally and return raw/calibrated touch forecasts.

    Target percentages are relative to the price at each forecast origin. Historical
    examples are limited to the same lookback and have fully resolved outcomes.
    Sampling every few minutes reduces runtime while retaining broad market coverage.
    """
    n = len(c)
    if not (len(o) == len(h) == len(l) == n):
        raise ValueError("OHLC series must have equal lengths")
    if max_horizon < 1 or lookback < 21 or sample_step < 1:
        raise ValueError("max_horizon, lookback, and sample_step must be positive and valid")

    atr = rolling_atr(h, l, c, n)
    mom = rolling_mom(o, c, n)
    future_up, future_down = rolling_horizon_extremes(h, l, c, n, max_horizon)
    keys = [(horizon, direction, target_pct)
            for horizon in range(1, max_horizon + 1)
            for direction in ("UP", "DOWN")
            for target_pct in target_pcts]
    forecasts = {key: [] for key in keys}
    weights = [math.exp(-DECAY * age) for age in range(lookback + 1)]

    for now in range(lookback, n - max_horizon, sample_step):
        current_atr = atr[now]
        current_mom = live_momentum_label(o, c, now)
        first_example = max(20, now - lookback)

        for horizon in range(1, max_horizon + 1):
            last_example = now - horizon
            all_total = 0.0
            matched_total = 0.0
            all_hits = {(direction, pct): 0.0 for direction in ("UP", "DOWN") for pct in target_pcts}
            matched_hits = dict.fromkeys(all_hits, 0.0)
            matched_count = 0

            for index in range(first_example, last_example + 1):
                weight = weights[now - index]
                all_total += weight
                is_match = (mom[index] == current_mom and current_atr > 0
                            and current_atr * 0.5 <= atr[index] <= current_atr * 2.0)
                if is_match:
                    matched_count += 1
                    matched_total += weight

                up_move = future_up[horizon][index]
                down_move = future_down[horizon][index]
                for pct in target_pcts:
                    if up_move >= pct:
                        all_hits[("UP", pct)] += weight
                        if is_match:
                            matched_hits[("UP", pct)] += weight
                    if down_move >= pct:
                        all_hits[("DOWN", pct)] += weight
                        if is_match:
                            matched_hits[("DOWN", pct)] += weight

            actual_up = future_up[horizon][now]
            actual_down = future_down[horizon][now]
            for direction in ("UP", "DOWN"):
                actual_move = actual_up if direction == "UP" else actual_down
                for pct in target_pcts:
                    overall = all_hits[(direction, pct)] / all_total * 100 if all_total else 0.0
                    if matched_count >= 15 and matched_total:
                        matched = matched_hits[(direction, pct)] / matched_total * 100
                        raw_prob = matched * 0.7 + overall * 0.3
                    else:
                        raw_prob = overall
                    key = (horizon, direction, pct)
                    forecasts[key].append({
                        "raw_probability_pct": raw_prob,
                        "probability_pct": calibrate_prob(raw_prob),
                        "reached": int(actual_move >= pct),
                    })

    return forecasts


def probability_metrics(records, probability_key):
    """Summarize probability forecasts with Brier skill and reliability bins."""
    if not records:
        return {"predictions": 0, "brier_score": None, "brier_skill_score": None,
                "observed_rate_pct": None, "mean_probability_pct": None, "bins": []}
    probabilities = [record[probability_key] / 100 for record in records]
    outcomes = [record["reached"] for record in records]
    event_rate = sum(outcomes) / len(outcomes)
    brier = sum((probability - outcome) ** 2
                for probability, outcome in zip(probabilities, outcomes)) / len(records)
    baseline_brier = event_rate * (1 - event_rate)
    bins = []
    for lower in range(0, 100, 10):
        grouped = [(probability, outcome)
                   for probability, outcome in zip(probabilities, outcomes)
                   if lower <= probability * 100 < lower + 10]
        if grouped:
            bins.append({
                "range_pct": [lower, lower + 10],
                "predictions": len(grouped),
                "mean_probability_pct": sum(p for p, _ in grouped) / len(grouped) * 100,
                "observed_rate_pct": sum(y for _, y in grouped) / len(grouped) * 100,
            })
    return {
        "predictions": len(records),
        "events": sum(outcomes),
        "observed_rate_pct": event_rate * 100,
        "mean_probability_pct": sum(probabilities) / len(probabilities) * 100,
        "brier_score": brier,
        "climatology_brier_score": baseline_brier,
        "brier_skill_score": 1 - brier / baseline_brier if baseline_brier else None,
        "bins": bins,
    }


def target_walkforward_main():
    """Evaluate deployed target-touch probabilities on older, non-overlapping windows.

    The calibration curve was derived from a window ending seven days ago, so these
    validation windows begin at least 21 days back to avoid that data and its neighbor.
    Run with: python backtest.py targetwalkforward
    """
    offsets = list(range(21, 71, 7))
    keys = [(horizon, direction, pct)
            for horizon in range(1, 16)
            for direction in ("UP", "DOWN")
            for pct in TARGET_PCTS]
    aggregate = {key: [] for key in keys}
    window_results = []

    for offset_days in offsets:
        end_time = int(time.time() * 1000) - offset_days * 24 * 3600 * 1000
        print(f"Fetching target-touch validation window ending {offset_days} days ago ...")
        raw = fetch_klines(WALKFORWARD_N_TARGET, end_time=end_time)
        if len(raw) <= 600:
            print(f"  skipped: only {len(raw)} candles")
            continue
        o, h, l, c, _ = build_series(raw)
        records = target_touch_records(o, h, l, c)
        window = {"offset_days": offset_days, "candles": len(raw), "metrics": {}}
        for key, forecasts in records.items():
            aggregate[key].extend(forecasts)
            horizon, direction, pct = key
            label = f"{horizon}m_{direction}_{pct:.2f}pct"
            window["metrics"][label] = {
                "raw": probability_metrics(forecasts, "raw_probability_pct"),
                "calibrated": probability_metrics(forecasts, "probability_pct"),
            }
        window_results.append(window)
        print(f"  scored {sum(len(items) for items in records.values())} forecasts")

    aggregate_metrics = {}
    for key, forecasts in aggregate.items():
        horizon, direction, pct = key
        label = f"{horizon}m_{direction}_{pct:.2f}pct"
        aggregate_metrics[label] = {
            "raw": probability_metrics(forecasts, "raw_probability_pct"),
            "calibrated": probability_metrics(forecasts, "probability_pct"),
            "windows_with_positive_calibrated_brier_skill": sum(
                window["metrics"][label]["calibrated"]["brier_skill_score"] is not None
                and window["metrics"][label]["calibrated"]["brier_skill_score"] > 0
                for window in window_results
            ),
        }

    report = {
        "symbol": SYMBOL,
        "evaluation": "causal target-touch probability; wick touch counts as reached",
        "target_move_percentages": TARGET_PCTS,
        "horizons_minutes": list(range(1, 16)),
        "lookback_candles": 500,
        "sample_step_minutes": 5,
        "excluded_recent_days": 21,
        "calibration_curve": "fixed deployed curve; unchanged during evaluation",
        "windows": window_results,
        "aggregate": aggregate_metrics,
        "interpretation": "Positive Brier skill means better than the same sample's constant event-rate forecast. This is historical evidence, not a guarantee of future accuracy.",
        "model_changed": False,
    }
    with open("target_touch_walkforward_results.json", "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print("Saved target-touch probability validation to target_touch_walkforward_results.json")


def rolling_future_extremes(h, l, c, n, horizon):
    """max favorable % move over the next `horizon` bars, from each index j."""
    max_up = [0.0] * n
    max_down = [0.0] * n
    for j in range(0, n - horizon):
        base = c[j]
        mu = md = 0.0
        for k in range(1, horizon + 1):
            mu = max(mu, (h[j + k] - base) / base * 100)
            md = max(md, (base - l[j + k]) / base * 100)
        max_up[j] = mu
        max_down[j] = md
    return max_up, max_down


def classify_confidence(strength):
    if strength >= 15:
        return "Strong"
    if strength >= 7:
        return "Moderate"
    return "Weak"


def can_cross(prob, aligned):
    if prob >= 30:
        return True
    if prob >= 15 and aligned:
        return True
    if prob > 0 and not aligned:
        return False
    return prob > 0


def bias_1min_backtest(o, c, mom_at, n):
    """Replicates the 'Bias: Bullish/Bearish/Neutral' label: unweighted,
    momentum-only match (no ATR filter), 1-minute-ahead horizon."""
    records = []  # (bullPct, label, actual_up, matched_samples)
    t0 = time.time()
    lo_T, hi_T = 40, n - 2
    total_T = hi_T - lo_T
    for T in range(lo_T, hi_T):
        trend = mom_at[T]
        up = total = 0
        for j in range(20, T):  # causal: need close[j+1] known -> j <= T-1
            if mom_at[j] != trend:
                continue
            total += 1
            if c[j + 1] > c[j]:
                up += 1
        bull_pct = (up / total * 100) if total > 0 else 50.0
        label = "Bullish" if bull_pct > 55 else "Bearish" if bull_pct < 45 else "Neutral"
        actual_up = 1 if c[T + 1] > c[T] else 0
        records.append((bull_pct, label, actual_up, total))
        if (T - lo_T) % 1000 == 0:
            pct_done = (T - lo_T) / total_T * 100
            print(f"  ... bias-1min {pct_done:5.1f}%  ({time.time()-t0:.1f}s elapsed)")
    print(f"Bias(1min) loop done in {time.time()-t0:.1f}s over {total_T} points\n")
    return records


def main():
    end_time = int(time.time() * 1000) - OUT_OF_SAMPLE_OFFSET_DAYS * 24 * 3600 * 1000
    print(f"Fetching ~{N_TARGET} 1m {SYMBOL} candles ending {OUT_OF_SAMPLE_OFFSET_DAYS} days ago "
          f"(OUT-OF-SAMPLE window, non-overlapping with the CALIBRATION_CURVE derivation window) ...")
    raw = fetch_klines(N_TARGET, end_time=end_time)
    n = len(raw)
    print(f"Got {n} candles  ({n/60/24:.2f} days)  "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[0][0]/1000))} UTC -> "
          f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(raw[-1][0]/1000))} UTC")

    o, h, l, c, v = build_series(raw)
    atr_at = rolling_atr(h, l, c, n)
    mom_at = rolling_mom(o, c, n)
    max_up, max_down = rolling_future_extremes(h, l, c, n, HORIZON)

    dir_records = []       # (predicted_upPct, direction, confidence, actual_up)
    cross_records = {}     # (pct, isUp) -> list of (prob, canCross, actual_reached)
    for pct in TARGET_PCTS:
        for is_up in (True, False):
            cross_records[(pct, is_up)] = []

    t0 = time.time()
    lo_T = 40
    hi_T = n - HORIZON - 1
    total_T = hi_T - lo_T

    for T in range(lo_T, hi_T):
        trend = mom_at[T]
        cur_atr = atr_at[T]

        w_up_all = w_total_all = 0.0
        w_up_best = w_total_best = 0.0
        matched_count = 0
        w_hits_all = {k: 0.0 for k in cross_records}
        w_hits_best = {k: 0.0 for k in cross_records}

        j_lo, j_hi = 20, T - HORIZON  # causal cutoff: outcome of j must already be known
        for j in range(j_lo, j_hi + 1):
            w = math.exp(-DECAY * (T - j))
            up_flag = 1.0 if c[j + HORIZON] > c[j] else 0.0

            w_total_all += w
            w_up_all += w * up_flag
            for key in cross_records:
                pct, is_up = key
                reached = 1.0 if (max_up[j] if is_up else max_down[j]) >= pct else 0.0
                w_hits_all[key] += w * reached

            same_vol = cur_atr > 0 and cur_atr * 0.5 <= atr_at[j] <= cur_atr * 2.0
            if mom_at[j] == trend and same_vol:
                matched_count += 1
                w_total_best += w
                w_up_best += w * up_flag
                for key in cross_records:
                    pct, is_up = key
                    reached = 1.0 if (max_up[j] if is_up else max_down[j]) >= pct else 0.0
                    w_hits_best[key] += w * reached

        overall_up_pct = (w_up_all / w_total_all * 100) if w_total_all > 0 else 50.0
        best_up_pct = (w_up_best / w_total_best * 100) if w_total_best > 0 else overall_up_pct
        base_up_pct = (best_up_pct * 0.7 + overall_up_pct * 0.3) if matched_count >= 15 else overall_up_pct

        recent_lo = max(0, T - 4)
        up_vol = sum(v[k] for k in range(recent_lo, T + 1) if c[k] >= o[k])
        down_vol = sum(v[k] for k in range(recent_lo, T + 1) if c[k] < o[k])
        vol_imb = (up_vol - down_vol) / (up_vol + down_vol) if (up_vol + down_vol) > 0 else 0.0

        up_pct = min(95.0, max(5.0, base_up_pct + vol_imb * 5))
        strength = abs(up_pct - 50)
        direction = "FLAT" if strength < 2 else ("UP" if up_pct > 50 else "DOWN")
        confidence = classify_confidence(strength)
        actual_up = 1 if c[T + HORIZON] > c[T] else 0
        dir_records.append((up_pct, direction, confidence, actual_up))

        for key in cross_records:
            pct, is_up = key
            overall_prob = (w_hits_all[key] / w_total_all * 100) if w_total_all > 0 else 0.0
            best_prob = (w_hits_best[key] / w_total_best * 100) if w_total_best > 0 else overall_prob
            raw_prob = (best_prob * 0.7 + overall_prob * 0.3) if matched_count >= 15 else overall_prob
            cal_prob = calibrate_prob(raw_prob)
            aligned = (is_up and trend == "up") or (not is_up and trend == "down")
            decision = can_cross(cal_prob, aligned)  # live code decides YES/NO on the calibrated value
            actual_reached = 1 if (max_up[T] if is_up else max_down[T]) >= pct else 0
            cross_records[key].append((raw_prob, cal_prob, decision, actual_reached))

        if (T - lo_T) % 500 == 0:
            pct_done = (T - lo_T) / total_T * 100
            print(f"  ... {pct_done:5.1f}%  ({time.time()-t0:.1f}s elapsed)")

    print(f"Backtest loop done in {time.time()-t0:.1f}s over {total_T} simulated points\n")

    # ---- Directional Bias report ----
    print("=" * 70)
    print("DIRECTIONAL BIAS (15 min) BACKTEST")
    print("=" * 70)
    calls = [r for r in dir_records if r[1] != "FLAT"]
    flat_rate = (len(dir_records) - len(calls)) / len(dir_records) * 100
    correct = sum(1 for p, d, conf, a in calls if (d == "UP") == (a == 1))
    acc = correct / len(calls) * 100 if calls else 0
    print(f"Total points: {len(dir_records)}   FLAT (no-call): {flat_rate:.1f}%")
    print(f"Directional accuracy (UP/DOWN calls only): {acc:.1f}%  ({correct}/{len(calls)})")

    print("\nAccuracy by confidence tier:")
    for tier in ("Strong", "Moderate", "Weak"):
        tier_calls = [r for r in calls if r[2] == tier]
        if not tier_calls:
            continue
        tcorrect = sum(1 for p, d, conf, a in tier_calls if (d == "UP") == (a == 1))
        print(f"  {tier:9s}: {tcorrect/len(tier_calls)*100:5.1f}%  (n={len(tier_calls)})")

    brier = sum((p / 100 - a) ** 2 for p, d, conf, a in dir_records) / len(dir_records)
    print(f"\nBrier score (upPct vs actual, 0=perfect, 0.25=coinflip-ish): {brier:.4f}")

    print("\nCalibration (predicted Up% bucket vs actual up-rate):")
    buckets = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
    for lo, hib in buckets:
        b = [r for r in dir_records if lo <= r[0] < hib]
        if not b:
            continue
        actual_rate = sum(a for p, d, conf, a in b) / len(b) * 100
        avg_pred = sum(p for p, d, conf, a in b) / len(b)
        print(f"  [{lo:3d}-{hib:3d}%) n={len(b):4d}  avg predicted={avg_pred:5.1f}%  actual up-rate={actual_rate:5.1f}%")

    # ---- Target-crossing calibration report (RAW vs CALIBRATED, out-of-sample) ----
    print("\n" + "=" * 70)
    print("TARGET-CROSSING PROBABILITY BACKTEST (breakoutProb) -- OUT-OF-SAMPLE")
    print("=" * 70)
    for pct in TARGET_PCTS:
        for is_up in (True, False):
            recs = cross_records[(pct, is_up)]
            dirname = "UP" if is_up else "DOWN"
            print(f"\n-- target {pct:.2f}% move {dirname} --")
            prob_buckets = [(0, 15), (15, 30), (30, 50), (50, 70), (70, 101)]
            print("   RAW probability (pre-calibration):")
            for lo, hib in prob_buckets:
                b = [r for r in recs if lo <= r[0] < hib]
                if not b:
                    continue
                actual_rate = sum(a for rp, cp, d, a in b) / len(b) * 100
                print(f"     [{lo:3d}-{hib:3d}%) n={len(b):4d}  actual hit-rate={actual_rate:5.1f}%")
            print("   CALIBRATED probability (what the dashboard now shows):")
            for lo, hib in prob_buckets:
                b = [r for r in recs if lo <= r[1] < hib]
                if not b:
                    continue
                actual_rate = sum(a for rp, cp, d, a in b) / len(b) * 100
                avg_cal = sum(cp for rp, cp, d, a in b) / len(b)
                print(f"     [{lo:3d}-{hib:3d}%) n={len(b):4d}  avg_calibrated={avg_cal:5.1f}%  actual hit-rate={actual_rate:5.1f}%")
            yes_recs = [r for r in recs if r[2]]
            no_recs = [r for r in recs if not r[2]]
            yes_hit = sum(a for rp, cp, d, a in yes_recs) / len(yes_recs) * 100 if yes_recs else 0
            no_hit = sum(a for rp, cp, d, a in no_recs) / len(no_recs) * 100 if no_recs else 0
            print(f"   Can Cross=YES actual hit-rate: {yes_hit:5.1f}%  (n={len(yes_recs)})")
            print(f"   Can Cross=NO  actual hit-rate: {no_hit:5.1f}%  (n={len(no_recs)})")

    # ---- Pooled: does the deployed CALIBRATION_CURVE hold up on fresh data? ----
    print("\n" + "=" * 70)
    print("POOLED: CALIBRATED PROB vs ACTUAL (out-of-sample validation of CALIBRATION_CURVE)")
    print("=" * 70)
    pooled = [r for recs in cross_records.values() for r in recs]  # (raw_prob, cal_prob, canCross, actual)
    edges = list(range(0, 101, 10))
    for i in range(len(edges) - 1):
        lo, hib = edges[i], edges[i + 1]
        b = [r for r in pooled if lo <= r[1] < hib] if hib < 100 else [r for r in pooled if lo <= r[1] <= hib]
        if not b:
            continue
        actual_rate = sum(a for rp, cp, d, a in b) / len(b) * 100
        avg_cal = sum(cp for rp, cp, d, a in b) / len(b)
        gap = avg_cal - actual_rate
        flag = "  <-- still off" if abs(gap) > 10 else ""
        print(f"  calibrated[{lo:3d}-{hib:3d}) n={len(b):5d}  avg_calibrated={avg_cal:5.1f}%  actual={actual_rate:5.1f}%  gap={gap:+5.1f}{flag}")

    # Also re-derive a fresh pooled raw curve for reference/comparison against the deployed one
    print("\nFor reference, freshly-derived RAW curve on this out-of-sample window:")
    curve_points = [(0.0, 0.0)]
    for i in range(len(edges) - 1):
        lo, hib = edges[i], edges[i + 1]
        b = [r for r in pooled if lo <= r[0] < hib] if hib < 100 else [r for r in pooled if lo <= r[0] <= hib]
        if not b:
            continue
        actual_rate = sum(a for rp, cp, d, a in b) / len(b) * 100
        avg_pred = sum(rp for rp, cp, d, a in b) / len(b)
        curve_points.append((avg_pred, actual_rate))
        print(f"  raw[{lo:3d}-{hib:3d}) n={len(b):5d}  avg_raw={avg_pred:5.1f}%  actual={actual_rate:5.1f}%")
    curve_points.append((100.0, curve_points[-1][1] if len(curve_points) > 1 else 100.0))
    print("[" + ", ".join(f"[{p:.1f},{a:.1f}]" for p, a in curve_points) + "]")

    # ---- Bias label (1 min ahead) report ----
    print("\n" + "=" * 70)
    print("BIAS LABEL (1 min ahead, unweighted, momentum-only match) BACKTEST")
    print("=" * 70)
    bias_records = bias_1min_backtest(o, c, mom_at, n)
    calls = [r for r in bias_records if r[1] != "Neutral"]
    neutral_rate = (len(bias_records) - len(calls)) / len(bias_records) * 100
    correct = sum(1 for bp, lbl, a, tot in calls if (lbl == "Bullish") == (a == 1))
    acc = correct / len(calls) * 100 if calls else 0
    print(f"Total points: {len(bias_records)}   Neutral (no-call): {neutral_rate:.1f}%")
    print(f"Directional accuracy (Bullish/Bearish calls only): {acc:.1f}%  ({correct}/{len(calls)})")

    brier = sum((bp / 100 - a) ** 2 for bp, lbl, a, tot in bias_records) / len(bias_records)
    print(f"Brier score: {brier:.4f}")

    print("\nCalibration (predicted bullPct bucket vs actual up-rate):")
    buckets = [(0, 40), (40, 45), (45, 55), (55, 60), (60, 101)]
    for lo, hib in buckets:
        b = [r for r in bias_records if lo <= r[0] < hib]
        if not b:
            continue
        actual_rate = sum(a for bp, lbl, a, tot in b) / len(b) * 100
        avg_pred = sum(bp for bp, lbl, a, tot in b) / len(b)
        avg_samples = sum(tot for bp, lbl, a, tot in b) / len(b)
        print(f"  [{lo:3d}-{hib:3d}%) n={len(b):5d}  avg predicted={avg_pred:5.1f}%  "
              f"actual up-rate={actual_rate:5.1f}%  avg matched samples={avg_samples:.0f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "combo":
        combo_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "rsicheck":
        rsi_check_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "walkforward":
        walkforward_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "targetwalkforward":
        target_walkforward_main()
    else:
        main()
