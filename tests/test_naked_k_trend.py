import json
import unittest

import numpy as np
import pandas as pd
from unittest.mock import patch

from naked_k_trend import analyze_trend, evaluate_entry, indicator_frame


def prices(closes, *, volume=100.0):
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": volume},
        index=pd.date_range("2025-01-01", periods=len(close)),
    )


class IndicatorTests(unittest.TestCase):
    def test_ema_is_sma_seeded_and_channels_exclude_current(self):
        result = indicator_frame(prices(range(10, 31)))
        self.assertEqual(result["ema20"].iloc[19], sum(range(10, 30)) / 20)
        self.assertEqual(result["ema20"].iloc[20], 19.5 + 2 / 21 * (30 - 19.5))
        self.assertEqual(result["channel_high"].iloc[20], 30.0)
        self.assertEqual(result["channel_low"].iloc[20], 9.0)

    def test_atr_seed_and_wilder_update(self):
        result = indicator_frame(prices([10] * 14 + [13]))
        self.assertEqual(result["atr"].iloc[13], 2.0)
        self.assertAlmostEqual(result["atr"].iloc[14], (2 * 13 + 4) / 14)

    def test_relative_volume_uses_prior_twenty_and_zero_volume_stays_null(self):
        data = prices(range(2, 32), volume=10.0)
        data.iloc[20, data.columns.get_loc("Volume")] = 20
        self.assertEqual(indicator_frame(data)["relative_volume"].iloc[20], 2.0)
        data["Volume"] = 0
        self.assertTrue(indicator_frame(data)["relative_volume"].isna().all())

    def test_flat_adx_is_zero_without_infinities(self):
        data = prices([10] * 40)
        data[["Open", "High", "Low", "Close"]] = 10.0
        result = indicator_frame(data)
        self.assertEqual(result["adx"].iloc[-1], 0.0)
        self.assertFalse(np.isinf(result.select_dtypes("number").to_numpy()).any())

    def test_one_way_market_adx_reaches_one_hundred(self):
        result = indicator_frame(prices(range(10, 50)))
        self.assertEqual(result["adx"].iloc[26], 100.0)

    def test_appending_future_rows_does_not_change_indicator_prefix(self):
        prefix = prices(np.linspace(10, 220, 205))
        longer = prices(np.linspace(10, 240, 220))
        longer.iloc[:205] = prefix.to_numpy()
        pd.testing.assert_frame_equal(indicator_frame(prefix), indicator_frame(longer).iloc[:205])

    def test_invalid_inputs_raise_clear_errors(self):
        cases = [prices([2, 1]).iloc[::-1],
                 prices([1, 2]).set_axis(pd.DatetimeIndex(["2025-01-01", "2025-01-01"])),
                 prices([1, 2]).assign(High=[2, 1]), prices([1, 2]).assign(Volume=[1, -1])]
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "OHLCV|DatetimeIndex"):
                    indicator_frame(data)


class TrendTests(unittest.TestCase):
    def test_short_history_displays_available_math_but_is_not_ready(self):
        signal = analyze_trend(prices(range(10, 70)))
        self.assertEqual(signal["status"], "insufficient_history")
        self.assertEqual(signal["direction"], "unknown")
        self.assertIsNotNone(signal["indicators"]["ema50"])
        self.assertFalse(signal["entry_candidate"])

    def test_uptrend_fresh_breakout_does_not_repeat(self):
        base = np.r_[np.linspace(50, 200, 184), np.full(20, 200.0)]
        first = prices(np.r_[base, 202])
        second = prices(np.r_[base, 202, 203])
        first_signal, second_signal = analyze_trend(first), analyze_trend(second)
        self.assertEqual(first_signal["direction"], "up")
        self.assertTrue(first_signal["breakout"])
        self.assertTrue(first_signal["entry_candidate"])
        self.assertFalse(second_signal["breakout"])
        self.assertFalse(second_signal["entry_candidate"])

    def test_fresh_breakout_payload_uses_json_native_booleans(self):
        close = np.r_[np.linspace(50, 200, 184), np.full(20, 200.0), 202]
        payload = analyze_trend(prices(close))

        self.assertTrue(payload["breakout"])
        self.assertTrue(payload["entry_candidate"])
        self.assertIs(type(payload["breakout"]), bool)
        self.assertIs(type(payload["entry_candidate"]), bool)
        self.assertIs(type(payload["exit_signal"]), bool)
        json.dumps(payload, allow_nan=False)

    def test_full_payload_has_no_nonfinite_json_numbers(self):
        payload = analyze_trend(prices([10] * 205, volume=0))
        json.dumps(payload, allow_nan=False)
        self.assertEqual(payload["direction"], "transition")
        self.assertEqual(payload["strength"], "weak")

    def test_overextended_breakout_is_not_an_entry_candidate(self):
        close = np.linspace(100, 200, 205)
        close[-21:-1] = 190
        data = prices(np.r_[close[:-1], 250])
        signal = analyze_trend(data)
        self.assertTrue(signal["breakout"])
        self.assertLess(signal["max_entry"], signal["close"])
        self.assertFalse(signal["entry_candidate"])

    def test_signal_uses_resistance_lower_edge_and_rejects_inside_zone(self):
        close = np.r_[np.linspace(50, 200, 184), np.full(20, 200.0), 202]
        zones = {"zones": [], "nearest_support": None,
                 "nearest_resistance": {"lower": 201.0, "upper": 205.0, "midpoint": 203.0},
                 "anchored_vwap": None}
        with patch("naked_k_trend.detect_price_zones", return_value=zones):
            signal = analyze_trend(prices(close))
        self.assertEqual(signal["resistance"], 201.0)
        self.assertFalse(signal["entry_candidate"])


class EntryTests(unittest.TestCase):
    def test_gap_above_max_entry_is_rejected(self):
        signal = {"entry_candidate": True, "max_entry": 100.0, "stop_loss": 90.0, "resistance": None}
        result = evaluate_entry(signal, 120.0)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["entry"], 120.0)

    def test_open_beyond_known_resistance_is_rejected(self):
        signal = {"entry_candidate": True, "max_entry": 130.0, "stop_loss": 90.0, "resistance": 110.0}
        result = evaluate_entry(signal, 120.0)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["resistance"], 110.0)

    def test_open_at_resistance_lower_edge_is_rejected(self):
        signal = {"entry_candidate": True, "max_entry": 130.0, "stop_loss": 90.0, "resistance": 110.0}
        result = evaluate_entry(signal, 110.0)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "insufficient_resistance_room")
