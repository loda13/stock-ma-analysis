import unittest
from zoneinfo import ZoneInfo
from naked_k_portfolio import classify_market
from unittest.mock import patch
import pandas as pd
import naked_k_analysis

class ClosedBarTests(unittest.TestCase):
    def test_korean_market_trims_unclosed_daily_bar_in_seoul_timezone(self):
        frame = pd.DataFrame(
            {
                "Open": [210000.0, 220000.0],
                "High": [215000.0, 225000.0],
                "Low": [205000.0, 218000.0],
                "Close": [212000.0, 219000.0],
                "Volume": [1000, 1200],
            },
            index=pd.to_datetime(["2026-07-08", "2026-07-09"]),
        )

        market = classify_market("000660.KS")
        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market=market,
            interval="1d",
            now=pd.Timestamp("2026-07-09 11:30:00", tz=ZoneInfo("Asia/Seoul")),
        )

        self.assertEqual(market, "kr")
        self.assertEqual(naked_k_analysis.market_timezone(market), ZoneInfo("Asia/Seoul"))
        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d"), "2026-07-08")

    def test_drop_incomplete_hk_daily_bar_before_close(self):
        frame = pd.DataFrame(
            {
                "Open": [420.0, 423.0],
                "High": [430.0, 425.0],
                "Low": [418.0, 420.0],
                "Close": [424.0, 423.6],
                "Volume": [1000, 900],
            },
            index=pd.to_datetime(["2026-06-26", "2026-06-29"]),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="hk",
            interval="1d",
            now=pd.Timestamp("2026-06-29 10:55:00", tz=ZoneInfo("Asia/Shanghai")),
        )

        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d"), "2026-06-26")

    def test_crypto_market_timezone_is_utc(self):
        self.assertEqual(naked_k_analysis.market_timezone("crypto"), ZoneInfo("UTC"))

    def test_crypto_periods_close_only_at_the_next_utc_boundary(self):
        cases = (
            (
                "1d",
                ["2026-08-19", "2026-08-20"],
                "2026-08-20 17:00:00",
                [pd.Timestamp("2026-08-19")],
            ),
            (
                "1wk",
                ["2026-08-10", "2026-08-17", "2026-08-20 09:29:24"],
                "2026-08-23 23:00:00",
                [pd.Timestamp("2026-08-10")],
            ),
            (
                "1mo",
                ["2026-07-01", "2026-08-01", "2026-08-20 09:29:24"],
                "2026-08-20 17:00:00",
                [pd.Timestamp("2026-07-01")],
            ),
        )
        for interval, index, now, expected in cases:
            with self.subTest(interval=interval):
                frame = pd.DataFrame(
                    {
                        "Open": range(len(index)),
                        "High": range(1, len(index) + 1),
                        "Low": range(len(index)),
                        "Close": range(1, len(index) + 1),
                        "Volume": [100] * len(index),
                    },
                    index=pd.to_datetime(index, format="mixed"),
                )

                trimmed = naked_k_analysis.trim_to_closed_bars(
                    frame,
                    market="crypto",
                    interval=interval,
                    now=pd.Timestamp(now, tz=ZoneInfo("UTC")),
                )

                self.assertEqual(trimmed.index.tolist(), expected)

    def test_load_ohlcv_applies_crypto_closed_bar_trimming(self):
        frame = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [2.0, 3.0],
                "Low": [0.5, 1.5],
                "Close": [1.5, 2.5],
                "Volume": [100.0, 200.0],
            },
            index=pd.to_datetime(["2026-08-19", "2026-08-20"]),
        )
        trim = naked_k_analysis.trim_to_closed_bars

        def trim_at_fixed_time(loaded, market, interval):
            return trim(
                loaded,
                market,
                interval,
                now=pd.Timestamp("2026-08-20 17:00:00", tz=ZoneInfo("UTC")),
            )

        with patch.object(naked_k_analysis.yf, "download", return_value=frame), patch.object(
            naked_k_analysis,
            "trim_to_closed_bars",
            side_effect=trim_at_fixed_time,
        ) as trim_call:
            loaded = naked_k_analysis.load_ohlcv("BTC-USD", interval="1d", period="18mo")

        self.assertEqual(loaded.index.tolist(), [pd.Timestamp("2026-08-19")])
        self.assertEqual(trim_call.call_args.kwargs, {"market": "crypto", "interval": "1d"})

    def test_drop_incomplete_hk_weekly_bar_during_current_week(self):
        frame = pd.DataFrame(
            {
                "Open": [410.0, 423.0],
                "High": [438.0, 425.0],
                "Low": [405.0, 420.0],
                "Close": [424.0, 423.6],
                "Volume": [5000, 900],
            },
            index=pd.to_datetime(["2026-06-26", "2026-06-29"]),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="hk",
            interval="1wk",
            now=pd.Timestamp("2026-06-29 10:55:00", tz=ZoneInfo("Asia/Shanghai")),
        )

        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d"), "2026-06-26")

    def test_drop_all_duplicate_current_week_rows(self):
        frame = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [2.0, 3.0, 4.0],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=pd.to_datetime(["2026-08-10", "2026-08-17", "2026-08-19 20:00"], format="mixed"),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="us",
            interval="1wk",
            now=pd.Timestamp("2026-08-20 10:00:00", tz=ZoneInfo("America/New_York")),
        )

        self.assertEqual(trimmed.index.tolist(), [pd.Timestamp("2026-08-10")])

    def test_drop_incomplete_monthly_bar_during_current_month(self):
        frame = pd.DataFrame(
            {
                "Open": [410.0, 423.0],
                "High": [438.0, 425.0],
                "Low": [405.0, 420.0],
                "Close": [424.0, 423.6],
                "Volume": [5000, 900],
            },
            index=pd.to_datetime(["2026-05-31", "2026-06-01"]),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="hk",
            interval="1mo",
            now=pd.Timestamp("2026-06-29 10:55:00", tz=ZoneInfo("Asia/Shanghai")),
        )

        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d"), "2026-05-31")

    def test_drop_all_duplicate_current_month_rows(self):
        frame = pd.DataFrame(
            {
                "Open": [1.0, 2.0, 3.0],
                "High": [2.0, 3.0, 4.0],
                "Low": [0.5, 1.5, 2.5],
                "Close": [1.5, 2.5, 3.5],
                "Volume": [100, 200, 300],
            },
            index=pd.to_datetime(["2026-07-01", "2026-08-01", "2026-08-19 20:00"], format="mixed"),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="us",
            interval="1mo",
            now=pd.Timestamp("2026-08-20 10:00:00", tz=ZoneInfo("America/New_York")),
        )

        self.assertEqual(trimmed.index.tolist(), [pd.Timestamp("2026-07-01")])

    def test_drop_zero_volume_latest_intraday_bar(self):
        frame = pd.DataFrame(
            {
                "Open": [420.0, 423.0],
                "High": [424.0, 425.0],
                "Low": [419.0, 422.0],
                "Close": [423.5, 424.0],
                "Volume": [1200, 0],
            },
            index=pd.to_datetime(["2026-06-30 06:00:00", "2026-06-30 07:00:00"]),
        )

        trimmed = naked_k_analysis.trim_to_closed_bars(
            frame,
            market="hk",
            interval="1h",
            now=pd.Timestamp("2026-06-30 15:20:00", tz=ZoneInfo("Asia/Shanghai")),
        )

        self.assertEqual(len(trimmed), 1)
        self.assertEqual(trimmed.index[-1].strftime("%Y-%m-%d %H:%M:%S"), "2026-06-30 06:00:00")


class AdjustmentConsistencyTests(unittest.TestCase):
    """Each timeframe runs its own fallback chain, so bases can diverge per run."""

    def _frame(self, adjustment, source="tencent"):
        frame = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )
        frame.attrs.update({"source": source, "adjustment": adjustment})
        return frame

    def test_audit_payload_records_the_adjustment_basis(self):
        payload = naked_k_analysis.build_data_audit_payload(
            "0700.HK", "1d", "18mo", self._frame("qfq")
        )

        self.assertEqual(payload["adjustment"], "qfq")

    def test_audit_payload_defaults_adjustment_to_unknown(self):
        frame = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        payload = naked_k_analysis.build_data_audit_payload("NVDA", "1d", "18mo", frame)

        self.assertEqual(payload["adjustment"], "unknown")

    def test_uniform_basis_produces_no_conflict(self):
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq"),
                "weekly": self._frame("qfq"),
                "monthly": self._frame("qfq"),
            }
        )

        self.assertIsNone(conflict)

    def test_mixed_basis_across_timeframes_is_reported(self):
        """The real failure: daily from Tencent qfq, weekly fell through to Yahoo."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("split_only", source="yahoo_chart"),
            }
        )

        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["bases"], {"daily": "qfq", "weekly": "split_only"})
        self.assertIn("日线", conflict["message"])
        self.assertIn("周线", conflict["message"])
        # The message must name the human-readable basis, not just the label.
        self.assertIn("前复权", conflict["message"])
        self.assertIn("仅拆股复权", conflict["message"])

    def test_unknown_from_two_different_sources_is_reported(self):
        """Two *different* silent sources are not evidence of a shared basis."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("unknown", source="westock"),
                "weekly": self._frame("unknown", source="some_other_provider"),
            }
        )

        self.assertIsNotNone(conflict)
        self.assertIn("未知", conflict["message"])

    def test_one_source_returning_two_bases_is_still_reported(self):
        """Same source is not a licence to skip the check.

        fetch_tencent_kline picks its label from whichever key answered, per
        request — so a symbol served `qfqday` but only a plain `week` yields qfq
        daily and split_only weekly, both tagged source='tencent'. Suppressing on
        source identity alone hid exactly the mismatch the labels exist to catch.
        """
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("qfq", source="tencent"),
                "monthly": self._frame("split_only", source="tencent"),
            }
        )

        self.assertIsNotNone(conflict)
        self.assertIn("月线", conflict["message"])

    def test_qfq_and_hfq_from_one_source_are_still_reported(self):
        """Both adjust fully, but anchor the price scale at opposite ends."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("hfq", source="tencent"),
            }
        )

        self.assertIsNotNone(conflict)

    def test_unknown_from_one_single_source_is_not_reported(self):
        """westock-data is first in the chain, so when present it serves all three.

        Its basis is undocumented and tagged `unknown`, but one source cannot
        disagree with itself — whatever the CLI returns, it returns the same thing
        for daily, weekly and monthly. Warning here would fire on every ticker on
        every run in exactly the environment where the primary source works, which
        is the failure mode this warning exists to avoid.
        """
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("unknown", source="westock"),
                "weekly": self._frame("unknown", source="westock"),
                "monthly": self._frame("unknown", source="westock"),
            }
        )

        self.assertIsNone(conflict)

    def test_unknown_mixed_with_a_known_basis_is_still_reported(self):
        """westock daily + Tencent qfq weekly: genuinely unverifiable, must warn."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("unknown", source="westock"),
                "weekly": self._frame("qfq", source="tencent"),
            }
        )

        self.assertIsNotNone(conflict)

    def test_intraday_alone_does_not_raise_a_conflict(self):
        """Intraday sits on an unobservable basis while daily/weekly are qfq.

        A-share 1h comes from Tencent's minute endpoint, which caps at 120 bars
        (~30 sessions), and HK 1h comes from Yahoo. Neither can reach past an
        ex-date within a 5d window, so the basis is unobservable — measured live,
        qfq and un-adjusted daily closes were identical to 0.0000% over it. The
        minute fetcher therefore reports `unknown`, and since `unknown` never
        compares equal, including intraday would warn on every A-share every run.
        Divergence only appears deeper in history (600519 hit 8.9% at 2y), which is
        why the structural timeframes below are still checked against each other.
        """
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("qfq", source="tencent"),
                "monthly": self._frame("qfq", source="tencent"),
                # What production actually produces now: A-share 1h on Tencent's
                # minute endpoint, self-labelled unknown.
                "intraday": self._frame("unknown", source="tencent"),
            }
        )

        self.assertIsNone(conflict)

    def test_hk_intraday_on_yahoo_also_raises_no_conflict(self):
        """HK cannot use Tencent's minute endpoint, so 1h stays on Yahoo."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("split_only", source="tencent"),
                "weekly": self._frame("split_only", source="tencent"),
                "monthly": self._frame("split_only", source="tencent"),
                "intraday": self._frame("split_only", source="yahoo_chart"),
            }
        )

        self.assertIsNone(conflict)

    def test_structural_timeframes_are_still_checked_against_each_other(self):
        """Monthly disagreeing with daily is the dangerous case and must warn."""
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("qfq", source="tencent"),
                "monthly": self._frame("split_only", source="yahoo_chart"),
                "intraday": self._frame("split_only", source="yahoo_chart"),
            }
        )

        self.assertIsNotNone(conflict)
        self.assertIn("月线", conflict["message"])
        # Intraday is out of scope, so it must not appear in the message either.
        self.assertNotIn("小时线", conflict["message"])
        self.assertNotIn("intraday", conflict["bases"])

    def test_missing_timeframes_are_skipped_not_flagged(self):
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq"),
                "weekly": self._frame("qfq"),
                "monthly": None,
                "intraday": None,
            }
        )

        self.assertIsNone(conflict)

    def test_empty_frames_are_skipped_not_flagged(self):
        empty = pd.DataFrame()
        empty.attrs["adjustment"] = "split_only"

        conflict = naked_k_analysis.detect_adjustment_conflict(
            {"daily": self._frame("qfq"), "weekly": self._frame("qfq"), "monthly": empty}
        )

        self.assertIsNone(conflict)

    def test_conflict_lists_the_source_behind_each_basis(self):
        conflict = naked_k_analysis.detect_adjustment_conflict(
            {
                "daily": self._frame("qfq", source="tencent"),
                "weekly": self._frame("split_only", source="yfinance"),
            }
        )

        self.assertEqual(
            conflict["sources"], {"daily": "tencent", "weekly": "yfinance"}
        )

    def test_report_renders_the_adjustment_warning_when_bases_diverge(self):
        conflict = {
            "bases": {"daily": "qfq", "weekly": "split_only"},
            "sources": {"daily": "tencent", "weekly": "yahoo_chart"},
            "message": "日线 前复权（tencent）与 周线 仅拆股复权（yahoo_chart）口径不一致",
        }

        line = naked_k_analysis.format_adjustment_warning(conflict)

        self.assertIn("⚠️", line)
        self.assertIn("复权口径", line)
        self.assertIn("口径不一致", line)

    def test_report_renders_nothing_when_there_is_no_conflict(self):
        self.assertEqual(naked_k_analysis.format_adjustment_warning(None), "")

    def _plan_frame(self):
        return pd.DataFrame(
            {
                "Open": [10.0, 11.0, 10.5, 12.0, 11.5, 14.0, 13.0, 15.0, 14.5, 17.0],
                "High": [12.0, 13.0, 12.5, 14.0, 13.5, 16.0, 15.0, 17.0, 16.5, 19.0],
                "Low": [8.0, 9.0, 8.5, 10.0, 9.5, 12.0, 11.0, 13.0, 12.5, 15.0],
                "Close": [9.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 16.0, 15.0, 18.5],
                "Volume": [1000, 1200, 950, 1300, 980, 1400, 1000, 1500, 1050, 1800],
            },
            index=pd.date_range("2026-06-01", periods=10, freq="D"),
        )

    def test_end_to_end_report_shows_the_warning_under_the_data_source_line(self):
        daily = self._plan_frame()
        weekly = daily.copy()
        report = naked_k_analysis.build_trade_plan("测试", "TEST", daily, weekly, previous=None)

        text = naked_k_analysis.format_report(
            "2026-06-12 16:00:00 CST",
            [report],
            naked_k_analysis.DEFAULT_JOURNAL_PATH,
            adjustment_conflicts={
                "TEST": {
                    "bases": {"daily": "qfq", "weekly": "split_only"},
                    "sources": {"daily": "tencent", "weekly": "yahoo_chart"},
                    "message": "日线 前复权（tencent）；周线 仅拆股复权（yahoo_chart） —— 口径不一致",
                }
            },
        )

        lines = text.splitlines()
        source_index = next(i for i, line in enumerate(lines) if line.startswith("- 数据源："))
        self.assertIn("复权口径警告", lines[source_index + 1])
        self.assertIn("⚠️", lines[source_index + 1])

    def test_end_to_end_report_omits_the_line_entirely_when_bases_agree(self):
        daily = self._plan_frame()
        weekly = daily.copy()
        report = naked_k_analysis.build_trade_plan("测试", "TEST", daily, weekly, previous=None)

        text = naked_k_analysis.format_report(
            "2026-06-12 16:00:00 CST",
            [report],
            naked_k_analysis.DEFAULT_JOURNAL_PATH,
            adjustment_conflicts={"TEST": None},
        )

        self.assertNotIn("复权口径警告", text)
        # No blank line smuggled in where the warning would have been.
        lines = text.splitlines()
        source_index = next(i for i, line in enumerate(lines) if line.startswith("- 数据源："))
        self.assertTrue(lines[source_index + 1].startswith("- 最新K线："))

    def test_end_to_end_report_tolerates_absent_conflict_mapping(self):
        """format_report is called without the kwarg in a dozen existing tests."""
        daily = self._plan_frame()
        weekly = daily.copy()
        report = naked_k_analysis.build_trade_plan("测试", "TEST", daily, weekly, previous=None)

        text = naked_k_analysis.format_report(
            "2026-06-12 16:00:00 CST", [report], naked_k_analysis.DEFAULT_JOURNAL_PATH
        )

        self.assertNotIn("复权口径警告", text)


class DataIntegrityTests(unittest.TestCase):
    def test_future_daily_bars_do_not_hide_incomplete_current_bar(self):
        f = pd.DataFrame({'Open': [10.] * 3, 'High': [11.] * 3, 'Low': [9.] * 3,
                          'Close': [10.] * 3, 'Volume': [100.] * 3},
                         index=pd.to_datetime(['2026-09-03', '2026-09-04', '2026-09-07']))
        result = naked_k_analysis.trim_to_closed_bars(f, 'hk', '1d',
                    now=pd.Timestamp('2026-09-04 12:00', tz='Asia/Hong_Kong'))
        self.assertEqual(list(result.index), [pd.Timestamp('2026-09-03')])

    def test_missing_volume_keeps_price_history_and_invalid_price_is_not_dropped(self):
        f = pd.DataFrame({'Open': [10., 10.], 'High': [11., 11.], 'Low': [9., 9.],
                          'Close': [10., 10.], 'Volume': [100., float('nan')]},
                         index=pd.to_datetime(['2026-08-03', '2026-08-04']))
        with patch.object(naked_k_analysis.yf, 'download', return_value=f):
            loaded = naked_k_analysis.load_ohlcv('TEST', '1d', '3y')
        self.assertEqual(len(loaded), 2)
        f.loc[f.index[-1], 'Close'] = float('nan')
        with patch.object(naked_k_analysis.yf, 'download', return_value=f):
            with self.assertRaises(ValueError):
                naked_k_analysis.load_ohlcv('TEST', '1d', '3y')
