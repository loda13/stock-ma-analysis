import unittest

import pandas as pd

from naked_k_zones import detect_price_zones


def frame(highs, lows, closes=None, volumes=None):
    closes = closes or [(high + low) / 2 for high, low in zip(highs, lows)]
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": volumes if volumes is not None else [100.0] * len(highs)},
        index=pd.date_range("2026-01-01", periods=len(highs)),
    )


class PriceZoneTests(unittest.TestCase):
    def test_only_confirmed_swings_become_zones(self):
        data = frame([10, 12, 11, 13, 12], [8, 9, 7, 10, 9])
        result = detect_price_zones(data, close=10.5, swing_window=1)
        self.assertEqual([zone["kind"] for zone in result["zones"]], ["resistance", "support", "resistance"])
        self.assertEqual(result["zones"][0]["confirmed_at"], "2026-01-03")
        self.assertEqual(result["zones"][1]["confirmed_at"], "2026-01-04")
        self.assertEqual(result["nearest_support"]["midpoint"], 8.0)
        self.assertEqual(result["nearest_resistance"]["midpoint"], 11.25)

    def test_pivot_appears_only_after_right_hand_confirmation(self):
        unconfirmed = frame([10, 12], [8, 9])
        confirmed = frame([10, 12, 11], [8, 9, 7])
        self.assertEqual(detect_price_zones(unconfirmed, swing_window=1)["zones"], [])
        self.assertEqual(detect_price_zones(confirmed, swing_window=1)["zones"][0]["confirmed_at"], "2026-01-03")

    def test_resistance_containing_close_is_relevant(self):
        data = frame([10, 12, 11], [8, 9, 7])
        resistance = detect_price_zones(data, close=11.5, swing_window=1)["nearest_resistance"]
        self.assertEqual(resistance["lower"], 10.5)
        self.assertEqual(resistance["upper"], 12.0)

    def test_anchored_vwap_uses_confirmed_swing_and_exposes_confirmation(self):
        data = frame([11, 10, 9, 11, 12, 13], [9, 8, 6, 8, 10, 11],
                     [10, 9, 8, 10, 11, 12], [1, 1, 2, 1, 1, 1])
        anchored = detect_price_zones(data, swing_window=1)["anchored_vwap"]
        expected = ((9 + 6 + 8) / 3 * 2 + 29 / 3 + 11 + 12) / 5
        self.assertEqual(anchored["anchor_date"], "2026-01-03")
        self.assertEqual(anchored["confirmed_at"], "2026-01-04")
        self.assertEqual(anchored["anchor_type"], "swing_low")
        self.assertAlmostEqual(anchored["value"], expected)
        self.assertEqual(anchored["source"], "structural_swing")

    def test_zero_volume_makes_anchored_vwap_unavailable(self):
        data = frame([11, 10, 9, 11, 12], [9, 8, 6, 8, 10], volumes=[0] * 5)
        self.assertIsNone(detect_price_zones(data, swing_window=1)["anchored_vwap"])

    def test_missing_volume_in_anchor_span_makes_vwap_unavailable(self):
        data = frame([11, 10, 9, 11, 12], [9, 8, 6, 8, 10], volumes=[1, 1, 2, float("nan"), 1])
        self.assertIsNone(detect_price_zones(data, swing_window=1)["anchored_vwap"])


if __name__ == "__main__":
    unittest.main()
