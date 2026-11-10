"""
Unit tests for the deterministic indicator/signal functions in backtest.py.
Run with: python -m unittest test_backtest -v   (stdlib unittest, no extra deps needed)

Network-dependent functions (fetch_klines) are tested with urllib mocked out --
no real HTTP calls are made by this suite. Functions that just orchestrate a full
run end-to-end (main, combo_main, rsi_check_main) are exercised via the project's
real-data backtests (see eth-dashboard-notes memory) and are not unit tested here.
"""
import statistics
import unittest
from unittest.mock import patch

import backtest as bt


class TestCalibrateProb(unittest.TestCase):
    def test_clamps_at_endpoints(self):
        self.assertEqual(bt.calibrate_prob(0), 0.0)
        self.assertEqual(bt.calibrate_prob(-10), 0.0)
        self.assertEqual(bt.calibrate_prob(100), 77.1)
        self.assertEqual(bt.calibrate_prob(150), 77.1)

    def test_exact_curve_point(self):
        self.assertEqual(bt.calibrate_prob(55.0), 55.3)

    def test_linear_interpolation(self):
        # Between (3.7, 7.6) and (15.2, 20.5)
        expected = 7.6 + (10 - 3.7) / (15.2 - 3.7) * (20.5 - 7.6)
        self.assertAlmostEqual(bt.calibrate_prob(10), expected, places=9)

    def test_monotonic_non_decreasing(self):
        prev = float("-inf")
        for raw in range(0, 101, 5):
            y = bt.calibrate_prob(raw)
            self.assertGreaterEqual(y, prev - 1e-9)
            prev = y


class TestBuildSeries(unittest.TestCase):
    def test_extracts_ohlcv_as_floats(self):
        raw = [[1690000000000, "100.5", "101.0", "99.5", "100.8", "12.34"]]
        o, h, l, c, v = bt.build_series(raw)
        self.assertEqual(o, [100.5])
        self.assertEqual(h, [101.0])
        self.assertEqual(l, [99.5])
        self.assertEqual(c, [100.8])
        self.assertEqual(v, [12.34])


class TestRollingATR(unittest.TestCase):
    def test_constant_true_range_ratio(self):
        n = 25
        c = [100.0] * n
        h = [101.0] * n  # (h-l)/c*100 = 1.0 for every bar
        l = [100.0] * n
        atr = bt.rolling_atr(h, l, c, n)
        for i in range(19):
            self.assertEqual(atr[i], 0.0)
        for i in range(19, n):
            self.assertAlmostEqual(atr[i], 1.0, places=9)


class TestRollingMom(unittest.TestCase):
    def test_up_down_flat_classification(self):
        n = 21
        o = [100.0] * n
        c = [100.0] * n
        c[20] = 100.1   # +0.1% vs o[0] -> up
        mom = bt.rolling_mom(o, c, n)
        self.assertEqual(mom[20], "up")
        for i in range(20):
            self.assertEqual(mom[i], "flat")  # below the i>=20 threshold

    def test_down_classification(self):
        n = 21
        o = [100.0] * n
        c = [100.0] * n
        c[20] = 99.9  # -0.1% -> down
        mom = bt.rolling_mom(o, c, n)
        self.assertEqual(mom[20], "down")

    def test_flat_when_within_threshold(self):
        n = 21
        o = [100.0] * n
        c = [100.0] * n  # 0% change -> flat
        mom = bt.rolling_mom(o, c, n)
        self.assertEqual(mom[20], "flat")

    def test_live_label_uses_exactly_last_20_candles(self):
        o = [100.0] * 21
        c = [100.0] * 21
        o[1] = 100.1
        c[20] = 100.04
        self.assertEqual(bt.rolling_mom(o, c, len(c))[20], "up")
        self.assertEqual(bt.live_momentum_label(o, c, 20), "down")


class TestRollingEMA(unittest.TestCase):
    def test_matches_hand_computed_values(self):
        series = [10.0, 20.0, 10.0, 20.0]
        ema = bt.rolling_ema(series, len(series), period=3)  # k = 0.5
        self.assertAlmostEqual(ema[0], 10.0, places=9)
        self.assertAlmostEqual(ema[1], 15.0, places=9)
        self.assertAlmostEqual(ema[2], 12.5, places=9)
        self.assertAlmostEqual(ema[3], 16.25, places=9)


class TestRollingRSI(unittest.TestCase):
    def test_matches_hand_computed_wilder_smoothing(self):
        # Same worked example as indicators.test.js (period=3):
        # closes 10,11,10,13,12,14 -> gains[1,0,3,0,2] losses[0,1,0,1,0]
        c = [10, 11, 10, 13, 12, 14]
        rsi = bt.rolling_rsi(c, len(c), period=3)
        self.assertAlmostEqual(rsi[3], 80.0, places=9)
        self.assertAlmostEqual(rsi[4], 61.53846153846154, places=9)
        self.assertAlmostEqual(rsi[5], 77.27272727272727, places=9)

    def test_strictly_rising_series_saturates_at_100(self):
        c = [100 + i for i in range(20)]
        rsi = bt.rolling_rsi(c, len(c), period=14)
        self.assertAlmostEqual(rsi[-1], 100.0, places=9)

    def test_strictly_falling_series_saturates_at_0(self):
        c = [200 - i for i in range(20)]
        rsi = bt.rolling_rsi(c, len(c), period=14)
        self.assertAlmostEqual(rsi[-1], 0.0, places=9)

    def test_stays_within_bounds_for_noisy_series(self):
        c = [100.0]
        seed = 42
        for _ in range(199):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            c.append(c[-1] + (seed / 0x7FFFFFFF - 0.5) * 4)
        rsi = bt.rolling_rsi(c, len(c), period=14)
        for v in rsi:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 100.0)


class TestRollingMACD(unittest.TestCase):
    def test_histogram_equals_macd_minus_signal(self):
        c = [100 + (i % 7) - (i % 3) for i in range(60)]
        macd_line, signal_line, hist = bt.rolling_macd(c, len(c), fast=12, slow=26, signal=9)
        n = len(c)
        self.assertEqual(len(macd_line), n)
        self.assertEqual(len(signal_line), n)
        self.assertEqual(len(hist), n)
        for i in range(n):
            self.assertAlmostEqual(hist[i], macd_line[i] - signal_line[i], places=9)

    def test_macd_line_equals_fast_minus_slow_ema(self):
        c = [100 + (i % 5) for i in range(40)]
        n = len(c)
        ema_fast = bt.rolling_ema(c, n, 12)
        ema_slow = bt.rolling_ema(c, n, 26)
        macd_line, _, _ = bt.rolling_macd(c, n, fast=12, slow=26, signal=9)
        for i in range(n):
            self.assertAlmostEqual(macd_line[i], ema_fast[i] - ema_slow[i], places=9)


class TestRollingBollinger(unittest.TestCase):
    def test_constant_series_has_zero_width_bands(self):
        n = 30
        c = [5.0] * n
        mid, upper, lower = bt.rolling_bollinger(c, n, period=20, mult=2.0)
        for i in range(19):
            self.assertEqual((mid[i], upper[i], lower[i]), (0.0, 0.0, 0.0))
        for i in range(19, n):
            self.assertAlmostEqual(mid[i], 5.0, places=9)
            self.assertAlmostEqual(upper[i], 5.0, places=9)
            self.assertAlmostEqual(lower[i], 5.0, places=9)

    def test_matches_population_stdev(self):
        n = 20
        c = [float(i + 1) for i in range(n)]  # 1..20
        mid, upper, lower = bt.rolling_bollinger(c, n, period=20, mult=2.0)
        expected_mean = statistics.mean(c)
        expected_sd = statistics.pstdev(c)
        self.assertAlmostEqual(mid[-1], expected_mean, places=9)
        self.assertAlmostEqual(upper[-1], expected_mean + 2 * expected_sd, places=9)
        self.assertAlmostEqual(lower[-1], expected_mean - 2 * expected_sd, places=9)


class TestRollingFutureExtremes(unittest.TestCase):
    def test_known_max_up_down_moves(self):
        # base=100 at j=0; over next 2 bars high touches 105 (+5%), low touches 97 (-3%)
        h = [100, 101, 105]
        l = [100, 99, 104]
        c = [100, 100, 100]
        n = 3
        max_up, max_down = bt.rolling_future_extremes(h, l, c, n, horizon=2)
        self.assertAlmostEqual(max_up[0], 5.0, places=9)
        self.assertAlmostEqual(max_down[0], 1.0, places=9)  # min low is 99 -> (100-99)/100*100=1%


class TestTargetTouchWalkforward(unittest.TestCase):
    def test_horizon_extremes_include_wicks_only_through_horizon(self):
        h = [100, 101, 104, 102]
        l = [100, 99, 98, 95]
        c = [100, 100, 100, 100]
        up, down = bt.rolling_horizon_extremes(h, l, c, len(c), 2)
        self.assertAlmostEqual(up[1][0], 1.0)
        self.assertAlmostEqual(up[2][0], 4.0)
        self.assertAlmostEqual(down[1][0], 1.0)
        self.assertAlmostEqual(down[2][0], 2.0)

    def test_probability_metrics_compare_to_event_rate_baseline(self):
        records = [
            {"probability_pct": 0, "reached": 0},
            {"probability_pct": 100, "reached": 1},
        ]
        metrics = bt.probability_metrics(records, "probability_pct")
        self.assertEqual(metrics["brier_score"], 0)
        self.assertEqual(metrics["brier_skill_score"], 1)
        self.assertEqual(metrics["observed_rate_pct"], 50)

    def test_future_bars_do_not_change_origin_probability(self):
        n = 504
        o = [100.0] * n
        h = [100.1] * n
        l = [99.9] * n
        c = [100.0] * n
        changed_h = h.copy()
        changed_h[501] = 100.3

        original = bt.target_touch_records(
            o, h, l, c, target_pcts=[0.15], max_horizon=2,
            lookback=500, sample_step=1000,
        )
        changed = bt.target_touch_records(
            o, changed_h, l, c, target_pcts=[0.15], max_horizon=2,
            lookback=500, sample_step=1000,
        )
        key = (1, "UP", 0.15)
        self.assertEqual(len(original[key]), 1)
        self.assertEqual(
            original[key][0]["raw_probability_pct"],
            changed[key][0]["raw_probability_pct"],
        )
        self.assertEqual(original[key][0]["reached"], 0)
        self.assertEqual(changed[key][0]["reached"], 1)


class TestClassifyConfidence(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(bt.classify_confidence(15), "Strong")
        self.assertEqual(bt.classify_confidence(14.999), "Moderate")
        self.assertEqual(bt.classify_confidence(7), "Moderate")
        self.assertEqual(bt.classify_confidence(6.999), "Weak")
        self.assertEqual(bt.classify_confidence(0), "Weak")


class TestCanCross(unittest.TestCase):
    def test_truth_table(self):
        self.assertTrue(bt.can_cross(35, False))       # >=30 regardless of alignment
        self.assertTrue(bt.can_cross(30, False))       # boundary >=30
        self.assertTrue(bt.can_cross(20, True))        # >=15 and aligned
        self.assertTrue(bt.can_cross(15, True))        # boundary >=15 and aligned
        self.assertFalse(bt.can_cross(20, False))      # >=15 but not aligned, <30
        self.assertFalse(bt.can_cross(5, False))       # small prob, not aligned
        self.assertFalse(bt.can_cross(0, False))
        self.assertFalse(bt.can_cross(0, True))


class TestWalkforwardMetrics(unittest.TestCase):
    def test_wilson_interval_contains_observed_accuracy(self):
        low, high = bt.wilson_interval(60, 100)
        self.assertLess(low, 60)
        self.assertGreater(high, 60)
        self.assertEqual(bt.wilson_interval(0, 0), [None, None])

    def test_strategy_metrics_includes_costs_and_confidence_interval(self):
        records = [
            ("UP", 1, 0.50),
            ("UP", 0, -0.20),
            ("DOWN", 0, 0.40),
            (None, 1, 0.10),
        ]
        metrics = bt.strategy_metrics(records, cost_pct=0.30)
        self.assertEqual(metrics["signals"], 3)
        self.assertAlmostEqual(metrics["accuracy_pct"], 2 / 3 * 100)
        self.assertEqual(metrics["up_signals"], 2)
        self.assertEqual(metrics["down_signals"], 1)
        self.assertAlmostEqual(metrics["cost_pct_round_trip"], 0.30)
        self.assertIsNotNone(metrics["accuracy_ci95_pct"])
        self.assertLess(metrics["net_expectancy_pct"], 0.50)

    def test_walkforward_records_gate_rsi_to_normal_regime(self):
        n = 50
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        v = [1.0] * n
        records, regimes = bt.build_walkforward_records(o, h, l, c, v, horizon=5)
        self.assertEqual(len(records["rsi"]), n - 40 - 5)
        self.assertEqual(len(records["rsi_normal"]), len(records["rsi"]))
        self.assertEqual(sum(len(values) for values in regimes.values()), len(records["rsi"]))


class TestRsiExtremeBacktest(unittest.TestCase):
    def test_signal_classification_and_actual_outcome(self):
        # warmup=2 for a short test series; horizon=1
        c = [100, 101, 102, 103, 90]
        rsi14 = [50, 25, 50, 75, 50]  # index2=oversold(UP), index3=overbought(DOWN)
        n = len(c)
        records = bt.rsi_extreme_backtest(c, n, rsi14, horizon=1, warmup=2, lo_th=30, hi_th=70)
        # loop runs T in range(2, n-1) = [2, 3]
        self.assertEqual(len(records), 2)
        sig2, actual2 = records[0]
        sig3, actual3 = records[1]
        self.assertIsNone(sig2)  # rsi14[2]=50 -> no signal
        self.assertEqual(actual2, 1)  # c[3]=103 > c[2]=102
        self.assertEqual(sig3, "DOWN")  # rsi14[3]=75 > hi_th
        self.assertEqual(actual3, 0)  # c[4]=90 < c[3]=103


class TestBollingerReversionBacktest(unittest.TestCase):
    def test_fades_closes_outside_bands(self):
        c = [100, 200, 100, 50, 100]
        upper = [0, 150, 150, 150, 150]  # index0 not warmed up (0)
        lower = [0, 80, 80, 80, 80]
        n = len(c)
        records = bt.bollinger_reversion_backtest(c, n, upper, lower, horizon=1, warmup=1)
        # loop runs T in range(1, n-1) = [1, 2, 3]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0][0], "DOWN")  # c[1]=200 > upper[1]=150
        self.assertIsNone(records[1][0])         # c[2]=100 within [80,150]
        self.assertEqual(records[2][0], "UP")    # c[3]=50 < lower[3]=80


class TestComboSignalBacktest(unittest.TestCase):
    # combo_signal_backtest has an internal `warmup = 40` bar, so every series below
    # is padded to length 42 and only index 40 is exercised (horizon=1 -> loop runs
    # over range(40, 41), i.e. exactly index 40).

    def test_all_three_bullish_fires_up(self):
        n = 42
        c = [100.0] * n
        c[41] = 105.0  # rises -> actual_up=1
        ema9 = [10.0] * n
        ema21 = [5.0] * n     # ema9 > ema21 -> 'up' trend
        rsi14 = [60.0] * n    # > 50 -> bullish
        hist = [1.0] * n      # > 0 -> bullish
        records = bt.combo_signal_backtest(c, n, ema9, ema21, rsi14, hist, horizon=1)
        self.assertEqual(records, [("UP", 1)])

    def test_all_three_bearish_fires_down(self):
        n = 42
        c = [100.0] * n
        c[41] = 95.0  # falls -> actual_up=0
        ema9 = [5.0] * n
        ema21 = [10.0] * n    # ema9 < ema21 -> 'down' trend
        rsi14 = [40.0] * n    # < 50 -> bearish
        hist = [-1.0] * n     # < 0 -> bearish
        records = bt.combo_signal_backtest(c, n, ema9, ema21, rsi14, hist, horizon=1)
        self.assertEqual(records, [("DOWN", 0)])

    def test_disagreement_yields_no_call(self):
        n = 42
        c = [100.0] * n
        c[41] = 105.0  # actual_up still computed regardless of signal
        ema9 = [10.0] * n
        ema21 = [5.0] * n     # 'up' trend ...
        rsi14 = [40.0] * n    # ... but rsi disagrees (bearish)
        hist = [1.0] * n
        records = bt.combo_signal_backtest(c, n, ema9, ema21, rsi14, hist, horizon=1)
        self.assertEqual(records, [(None, 1)])


class TestFetchKlines(unittest.TestCase):
    def test_paginates_backwards_until_total_reached(self):
        batch_new = [[i, "1", "1", "1", "1", "1"] for i in range(1000, 2000)]  # newest 1000
        batch_old = [[i, "1", "1", "1", "1", "1"] for i in range(0, 1000)]     # older 1000

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                import json as _json
                return _json.dumps(self._payload).encode()

        responses = [FakeResponse(batch_new), FakeResponse(batch_old)]

        with patch("backtest.urllib.request.urlopen", side_effect=responses), \
             patch("backtest.time.sleep", return_value=None):
            result = bt.fetch_klines(1500)

        self.assertEqual(len(result), 1500)
        # Most recent 1500 of the combined 2000 candles (open times 500..1999)
        self.assertEqual(result[0][0], 500)
        self.assertEqual(result[-1][0], 1999)


if __name__ == "__main__":
    unittest.main()
