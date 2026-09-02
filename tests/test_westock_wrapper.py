import os
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import westock_wrapper


class WestockWrapperTests(unittest.TestCase):
    def test_wrapper_no_longer_imports_legacy_stock_analysis_package(self):
        source = inspect.getsource(westock_wrapper)

        self.assertNotIn("stock_analysis", source)

    def test_convert_ticker_normalizes_common_markets_locally(self):
        self.assertEqual(westock_wrapper.convert_ticker("0700.HK"), "hk00700")
        self.assertEqual(westock_wrapper.convert_ticker("600703.SS"), "sh600703")
        self.assertEqual(westock_wrapper.convert_ticker("001391.SZ"), "sz001391")
        self.assertEqual(westock_wrapper.convert_ticker("000660.KS"), "kr000660")
        self.assertEqual(westock_wrapper.convert_ticker("NVDA"), "usNVDA")
        self.assertEqual(westock_wrapper.convert_ticker("688256.SS"), "sh688256")
        self.assertEqual(westock_wrapper.convert_ticker("430139.BJ"), "bj430139")
        self.assertEqual(westock_wrapper.convert_ticker("9992.HK"), "hk09992")

    def test_convert_ticker_needs_no_hand_maintained_lookup_table(self):
        """The old TICKER_MAP's 11 entries all agreed with the rule below it.

        Keeping it implied new symbols had to be registered first, which was
        never true. Asserted so it does not grow back.
        """
        self.assertFalse(hasattr(westock_wrapper, "TICKER_MAP"))

    def test_convert_ticker_is_case_and_whitespace_insensitive(self):
        self.assertEqual(westock_wrapper.convert_ticker(" 0700.hk "), "hk00700")
        self.assertEqual(westock_wrapper.convert_ticker("nvda"), "usNVDA")

    def test_uses_env_script_path(self):
        with patch.dict(os.environ, {"WESTOCK_DATA_SCRIPT": "/tmp/westock.js"}):
            cmd = westock_wrapper.build_westock_command("NVDA", "day", 10)

        self.assertEqual(cmd[1], "/tmp/westock.js")
        self.assertEqual(cmd[-3:], ["usNVDA", "day", "10"])

    def test_download_falls_back_to_yfinance_when_westock_empty(self):
        fallback_df = SimpleNamespace(empty=False)
        empty_df = SimpleNamespace(empty=True)

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=empty_df
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_yfinance", return_value=fallback_df
        ) as fallback:
            result = westock_wrapper.download("NVDA", period="1y", interval="1d")

        self.assertIs(result, fallback_df)
        fallback.assert_called_once()

    def test_download_uses_yahoo_chart_before_yfinance_when_tencent_empty(self):
        empty_df = pd.DataFrame()
        yahoo_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=empty_df
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=yahoo_df), patch.object(
            westock_wrapper, "fetch_yfinance"
        ) as fallback:
            result = westock_wrapper.download("NVDA", period="1y", interval="1d")

        self.assertIs(result, yahoo_df)
        fallback.assert_not_called()
        self.assertEqual(result.attrs["source"], "yahoo_chart")
        self.assertEqual(result.attrs["rows"], 1)
        self.assertEqual(result.attrs["latest"], "2026-06-01")

    def test_download_skips_tencent_kline_for_us_ticker(self):
        empty_df = pd.DataFrame()
        yahoo_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline"
        ) as tencent, patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=yahoo_df):
            result = westock_wrapper.download("NVDA", period="1y", interval="1d")

        self.assertIs(result, yahoo_df)
        tencent.assert_not_called()

    def test_fetch_tencent_kline_parses_hk_daily_rows(self):
        payload = {
            "code": 0,
            "data": {
                "hk00700": {
                    "day": [
                        ["2026-05-29", "430.000", "438.000", "439.000", "429.000", "12345.000", {}],
                        ["2026-06-01", "438.000", "440.000", "445.000", "437.000", "23456.000", {}],
                    ]
                }
            },
        }

        response = SimpleNamespace(text=json.dumps(payload), raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_kline("0700.HK", period="day", limit=2)

        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(df.index[0].strftime("%Y-%m-%d"), "2026-05-29")
        self.assertEqual(float(df.iloc[-1]["Close"]), 440.0)
        self.assertEqual(float(df.iloc[-1]["Volume"]), 23456.0)

    def test_fetch_tencent_kline_reads_qfq_key_for_a_shares(self):
        """A-shares answer a qfq request under ``qfqday``, not ``day``.

        Verified live: sh688256 returns ``day`` first_close=1009.450 but
        ``qfqday`` first_close=676.477 over the same 90 bars. Reading the wrong
        key yields un-adjusted prices with synthetic ex-rights gaps.
        """
        payload = {
            "code": 0,
            "data": {
                "sh688256": {
                    "qfqday": [
                        ["2026-05-29", "670.000", "676.477", "680.000", "665.000", "1000.000", {}],
                    ]
                }
            },
        }

        response = SimpleNamespace(text=json.dumps(payload), raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_kline("688256.SS", period="day", limit=1)

        self.assertFalse(df.empty)
        self.assertEqual(float(df.iloc[-1]["Close"]), 676.477)
        self.assertEqual(df.attrs["adjustment"], "qfq")

    def test_fetch_tencent_kline_prefers_qfq_key_over_unadjusted_day(self):
        """When both keys are present the adjusted series must win.

        Defensive only: probed live across 8 symbols x day/week/month and no
        response carried both keys, so this is not covering a real regression.
        Pinned anyway because the basis label is derived from which key answered.
        """
        payload = {
            "code": 0,
            "data": {
                "sh600519": {
                    "day": [
                        ["2026-05-29", "1420.000", "1420.000", "1430.000", "1410.000", "500.000", {}],
                    ],
                    "qfqday": [
                        ["2026-05-29", "1391.976", "1391.976", "1400.000", "1385.000", "500.000", {}],
                    ],
                }
            },
        }

        response = SimpleNamespace(text=json.dumps(payload), raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_kline("600519.SS", period="day", limit=1)

        self.assertEqual(float(df.iloc[-1]["Close"]), 1391.976)
        self.assertEqual(df.attrs["adjustment"], "qfq")

    def test_fetch_tencent_kline_marks_hk_day_key_as_split_only(self):
        """HK answers under plain ``day`` and that series is dividend-raw.

        Verified live twice over: hk00700 returns byte-identical rows for
        ``''``/``qfq``/``hfq`` (the mode is ignored), and across 489 bars spanning
        two dividends it tracks Yahoo's un-adjusted close to a 0.0027% mean while
        diverging from adjclose by 1.33% on 430 bars. So the label follows what
        the data *is*, not what the request asked for.
        """
        payload = {
            "code": 0,
            "data": {
                "hk00700": {
                    "day": [
                        ["2026-06-01", "438.000", "440.000", "445.000", "437.000", "23456.000", {}],
                    ]
                }
            },
        }

        response = SimpleNamespace(text=json.dumps(payload), raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_kline("0700.HK", period="day", limit=1)

        self.assertEqual(df.attrs["adjustment"], "split_only")

    def test_fetch_yahoo_chart_reports_split_only_adjustment(self):
        """Yahoo's quote OHLC is split-adjusted but not dividend-adjusted."""
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1780272000],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [432.4],
                                    "high": [442.0],
                                    "low": [430.0],
                                    "close": [438.4],
                                    "volume": [21445870],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        response = SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_yahoo_chart("0700.HK", period="5d", interval="1d")

        self.assertEqual(df.attrs["adjustment"], "split_only")

    def test_download_propagates_adjustment_from_the_winning_source(self):
        empty_df = pd.DataFrame()
        tencent_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )
        tencent_df.attrs["adjustment"] = "qfq"

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=tencent_df
        ), patch.object(westock_wrapper, "fetch_yfinance"):
            result = westock_wrapper.download("0700.HK", period="1y", interval="1d")

        self.assertEqual(result.attrs["source"], "tencent")
        self.assertEqual(result.attrs["adjustment"], "qfq")

    def test_download_labels_unknown_adjustment_when_source_is_silent(self):
        """A source that never set the field must not inherit a stale label."""
        empty_df = pd.DataFrame()
        bare_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=empty_df
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_yfinance", return_value=bare_df
        ):
            result = westock_wrapper.download("NVDA", period="1y", interval="1d")

        self.assertEqual(result.attrs["adjustment"], "unknown")

    def test_fetch_tencent_kline_uses_backup_domain_when_primary_fails(self):
        payload = {
            "code": 0,
            "data": {
                "hk00700": {
                    "day": [
                        ["2026-06-01", "438.000", "440.000", "445.000", "437.000", "23456.000", {}],
                    ]
                }
            },
        }

        response = SimpleNamespace(text=json.dumps(payload), raise_for_status=lambda: None)
        with patch("requests.get", side_effect=[RuntimeError("dns"), response]) as get:
            df = westock_wrapper.fetch_tencent_kline("0700.HK", period="day", limit=1)

        self.assertFalse(df.empty)
        self.assertEqual(get.call_count, 2)

    def test_fetch_yahoo_chart_parses_daily_rows(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1780272000, 1780358400],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [432.4, 438.0],
                                    "high": [442.0, 445.0],
                                    "low": [430.0, 437.0],
                                    "close": [438.4, 440.0],
                                    "volume": [21445870, 23456000],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        response = SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)
        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_yahoo_chart("0700.HK", period="5d", interval="1d")

        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(float(df.iloc[-1]["Close"]), 440.0)
        self.assertEqual(float(df.iloc[-1]["Volume"]), 23456000.0)

    def test_fetch_kline_skips_missing_default_westock_script(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.path.exists", return_value=False), patch(
            "subprocess.run"
        ) as run:
            df = westock_wrapper.fetch_kline("0700.HK", period="day", limit=10)

        self.assertTrue(df.empty)
        run.assert_not_called()

    def test_download_uses_tencent_before_yfinance_when_westock_empty(self):
        empty_df = pd.DataFrame()
        tencent_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=tencent_df
        ), patch.object(westock_wrapper, "fetch_yfinance") as fallback:
            result = westock_wrapper.download("0700.HK", period="1y", interval="1d")

        self.assertIs(result, tencent_df)
        self.assertEqual(result.attrs["source"], "tencent")
        fallback.assert_not_called()

    def test_download_uses_tencent_hourly_for_hk_1h(self):
        empty_df = pd.DataFrame()
        hourly_df = pd.DataFrame(
            {"Open": [1.0] * 180, "High": [2.0] * 180, "Low": [0.5] * 180, "Close": [1.5] * 180, "Volume": [100.0] * 180},
            index=pd.date_range("2026-05-01 10:00", periods=180, freq="h"),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=hourly_df
        ) as tencent, patch.object(westock_wrapper, "fetch_yahoo_chart") as yahoo:
            result = westock_wrapper.download("0700.HK", period="60d", interval="1h")

        self.assertIs(result, hourly_df)
        self.assertEqual(tencent.call_args.args[1], "m60")
        yahoo.assert_not_called()

    def test_fetch_yfinance_reports_split_only_adjustment(self):
        """``auto_adjust=False`` keeps Yahoo's split-adjusted, dividend-raw OHLC."""
        frame = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01"]),
        )

        with patch("yfinance.download", return_value=frame) as download:
            result = westock_wrapper.fetch_yfinance("NVDA", period="1y", interval="1d")

        self.assertEqual(result.attrs["adjustment"], "split_only")
        self.assertFalse(download.call_args.kwargs["auto_adjust"])

    def test_fetch_yfinance_flattens_single_ticker_multiindex_columns(self):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        for columns in (
            pd.MultiIndex.from_product([fields, ["1810.HK"]]),
            pd.MultiIndex.from_product([["1810.HK"], fields]),
        ):
            with self.subTest(columns=columns.tolist()):
                frame = pd.DataFrame(
                    [[1.0, 2.0, 0.5, 1.5, 100.0]],
                    columns=columns,
                    index=pd.to_datetime(["2026-09-02 10:30:00+08:00"]),
                )

                with patch("yfinance.download", return_value=frame):
                    result = westock_wrapper.fetch_yfinance(
                        "1810.HK", period="5d", interval="1h"
                    )

                self.assertEqual(list(result.columns), fields)
                self.assertEqual(float(result.iloc[-1]["Volume"]), 100.0)


class TencentMinuteKlineTests(unittest.TestCase):
    """Minute bars live on a different endpoint than day/week/month.

    `fqkline/get` serves no minute data at all: verified live, hk00700 m60/m30/m15
    return exactly 1 row and sh688256 returns 0 with code=1. Minute bars come from
    `kline/mkline`, which production never queried — so A-share intraday could only
    ever arrive from Yahoo.
    """

    def _payload(self, symbol, rows):
        return {"code": 0, "data": {symbol: {"m60": rows}}}

    def test_parses_open_close_high_low_volume_order(self):
        """Row layout is [time, open, close, high, low, volume] — close before high.

        Confirmed live on sh688256: field 3 was the row max and field 4 the row min
        on all 10 sampled bars, which only holds for this ordering.
        """
        rows = [
            ["202608061400", "1145.01", "1150.00", "1158.00", "1141.69", "1554234.00", {}],
            ["202608061500", "1150.00", "1168.15", "1168.15", "1141.01", "2204478.00", {}],
        ]
        response = SimpleNamespace(
            text=json.dumps(self._payload("sh688256", rows)), raise_for_status=lambda: None
        )

        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_minute_kline("688256.SS", "m60", 120)

        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        latest = df.iloc[-1]
        self.assertEqual(float(latest["Open"]), 1150.00)
        self.assertEqual(float(latest["Close"]), 1168.15)
        self.assertEqual(float(latest["High"]), 1168.15)
        self.assertEqual(float(latest["Low"]), 1141.01)
        self.assertEqual(float(latest["Volume"]), 2204478.0)

    def test_converts_beijing_timestamps_to_naive_utc(self):
        """mkline returns naive Beijing time; Yahoo intraday is naive UTC.

        Verified live: Yahoo's 0700.HK 1h index ends at 06:07 for a session that
        closes 16:00 Beijing, i.e. UTC with no tzinfo. Leaving mkline in Beijing
        would put the two sources 8h apart on the same bar. Normalising to naive
        UTC keeps one convention.
        """
        rows = [["202608061400", "1145.01", "1150.00", "1158.00", "1141.69", "1554234.00", {}]]
        response = SimpleNamespace(
            text=json.dumps(self._payload("sh688256", rows)), raise_for_status=lambda: None
        )

        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_minute_kline("688256.SS", "m60", 120)

        # 14:00 Beijing == 06:00 UTC.
        self.assertIsNone(df.index.tz)
        self.assertEqual(df.index[-1].strftime("%Y-%m-%d %H:%M"), "2026-08-06 06:00")

    def test_labels_adjustment_unknown_because_the_window_cannot_reach_an_ex_date(self):
        """mkline caps at 120 bars ~= 30 A-share days, so its basis is unobservable.

        Live check over that reachable window: qfq and un-adjusted daily closes were
        identical to 0.0000%, so no evidence distinguishes them. Claiming either
        label would be a guess, and a wrong label warns on every run.
        """
        rows = [["202608061400", "1145.01", "1150.00", "1158.00", "1141.69", "1554234.00", {}]]
        response = SimpleNamespace(
            text=json.dumps(self._payload("sh688256", rows)), raise_for_status=lambda: None
        )

        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_minute_kline("688256.SS", "m60", 120)

        self.assertEqual(df.attrs["adjustment"], westock_wrapper.ADJUSTMENT_UNKNOWN)

    def test_returns_empty_for_hk_rather_than_pretending(self):
        """HK is not served: mkline answers code=-1 for hk00700 on m60/m30/m15."""
        response = SimpleNamespace(
            text=json.dumps({"code": -1, "data": {}}), raise_for_status=lambda: None
        )

        with patch("requests.get", return_value=response):
            df = westock_wrapper.fetch_tencent_minute_kline("0700.HK", "m60", 120)

        self.assertTrue(df.empty)

    def test_survives_a_transport_failure(self):
        with patch("requests.get", side_effect=RuntimeError("dns")):
            df = westock_wrapper.fetch_tencent_minute_kline("688256.SS", "m60", 120)

        self.assertTrue(df.empty)

    def test_download_routes_1h_to_the_minute_endpoint_not_fqkline(self):
        """The regression that hid this: 1h asked fqkline/get, which has no m60."""
        parsed = pd.DataFrame(
            {"Open": [1.0] * 20, "High": [2.0] * 20, "Low": [0.5] * 20, "Close": [1.5] * 20, "Volume": [9.0] * 20},
            index=pd.date_range("2026-08-03 01:30", periods=20, freq="h"),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_kline"
        ) as daily_endpoint, patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=parsed
        ) as minute_endpoint, patch.object(
            westock_wrapper, "fetch_yahoo_chart"
        ) as yahoo:
            result = westock_wrapper.download("688256.SS", period="5d", interval="1h")

        self.assertIs(result, parsed)
        self.assertEqual(result.attrs["source"], "tencent")
        minute_endpoint.assert_called_once()
        daily_endpoint.assert_not_called()
        yahoo.assert_not_called()

    def test_download_still_uses_fqkline_for_daily(self):
        parsed = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [9.0]},
            index=pd.to_datetime(["2026-08-10"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=parsed
        ) as daily_endpoint, patch.object(
            westock_wrapper, "fetch_tencent_minute_kline"
        ) as minute_endpoint:
            westock_wrapper.download("688256.SS", period="18mo", interval="1d")

        daily_endpoint.assert_called_once()
        minute_endpoint.assert_not_called()

    def test_hk_1h_falls_through_to_yahoo(self):
        """A-shares get Tencent intraday, HK cannot — so HK must still reach Yahoo."""
        yahoo_df = pd.DataFrame(
            {"Open": [1.0] * 30, "High": [2.0] * 30, "Low": [0.5] * 30, "Close": [1.5] * 30, "Volume": [9.0] * 30},
            index=pd.date_range("2026-08-03 01:30", periods=30, freq="h"),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=pd.DataFrame()
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=yahoo_df) as yahoo:
            result = westock_wrapper.download("0700.HK", period="5d", interval="1h")

        self.assertIs(result, yahoo_df)
        self.assertEqual(result.attrs["source"], "yahoo_chart")
        yahoo.assert_called_once()

    def test_tiny_minute_frame_falls_back_to_yahoo_via_the_scaled_gate(self):
        """The scaled row gate applies to the minute endpoint too, not just fqkline."""
        empty_df = pd.DataFrame()
        tiny_minute_df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]},
            index=pd.to_datetime(["2026-06-01 10:00"]),
        )
        # 60d * 4 bars/day / 2 = 120, capped at 119, so 119+ passes the gate.
        yahoo_df = pd.DataFrame(
            {"Open": [1.0] * 119, "High": [2.0] * 119, "Low": [0.5] * 119, "Close": [1.5] * 119, "Volume": [100.0] * 119},
            index=pd.date_range("2026-05-01 10:00", periods=119, freq="h"),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=empty_df), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=tiny_minute_df
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=yahoo_df) as yahoo:
            result = westock_wrapper.download("688256.SS", period="60d", interval="1h")

        self.assertIs(result, yahoo_df)
        yahoo.assert_called_once()


class IntradayRowGateTests(unittest.TestCase):
    """MIN_INTRADAY_ROWS=120 was unsatisfiable for the period production requests.

    A-shares trade 4h/day and HK ~5.5h, so a 5d window holds at most 20 and ~30
    hourly bars. The gate demanded 120 and was applied only to westock/Tencent,
    never to Yahoo — so the two China sources were guaranteed to be discarded for
    intraday while a 5-row Yahoo frame was accepted. The threshold must scale with
    the requested window.
    """

    def _accepted(self, rows, ticker="688256.SS", period="5d"):
        frame = pd.DataFrame(
            {"Open": [1.0] * rows, "High": [2.0] * rows, "Low": [0.5] * rows,
             "Close": [1.5] * rows, "Volume": [9.0] * rows},
            index=pd.date_range("2026-08-03 01:30", periods=rows, freq="h"),
        )
        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=frame
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_yfinance", return_value=pd.DataFrame()
        ):
            result = westock_wrapper.download(ticker, period=period, interval="1h")
        return result.attrs.get("source") == "tencent"

    def test_a_realistic_5d_a_share_frame_is_accepted(self):
        """20 bars is the physical maximum for 5d A-share; it must not be rejected."""
        self.assertTrue(self._accepted(20))

    def test_a_realistic_5d_hk_frame_is_accepted(self):
        self.assertTrue(self._accepted(30, ticker="0700.HK"))

    def test_a_single_stub_bar_is_still_rejected(self):
        """The gate's real job: reject the 1-row stub fqkline/get returns."""
        self.assertFalse(self._accepted(1))

    def test_threshold_scales_with_the_requested_window(self):
        expected_minimum = westock_wrapper.min_intraday_rows("60d")
        self.assertGreater(expected_minimum, westock_wrapper.min_intraday_rows("5d"))

    def test_threshold_never_exceeds_what_the_minute_endpoint_can_return(self):
        """mkline hard-caps at 120 bars, so a gate above that rejects a full frame.

        For period='60d' the scaled threshold landed on exactly 120 — equal to the
        cap, leaving zero margin. One holiday in the window and Tencent would be
        discarded again, which is the bug this gate was rewritten to remove.
        """
        for period in ("5d", "60d", "18mo", "5y", "10y", "max"):
            with self.subTest(period=period):
                self.assertLess(
                    westock_wrapper.min_intraday_rows(period),
                    westock_wrapper.TENCENT_MINUTE_MAX_ROWS,
                )

    def test_threshold_tolerates_a_malformed_period(self):
        """download() is a public shim; a bad period must not raise from the gate."""
        for period in (None, "", "bogus", "2wk", 5):
            with self.subTest(period=period):
                self.assertGreaterEqual(westock_wrapper.min_intraday_rows(period), 2)

    def test_threshold_for_5d_is_reachable_in_a_5d_session_window(self):
        """Whatever the rule, it must be satisfiable by real 5d data."""
        self.assertLessEqual(westock_wrapper.min_intraday_rows("5d"), 20)

    def test_gate_applies_to_yahoo_too(self):
        """Yahoo bypassed the gate entirely, so a 1-row stub was accepted as valid."""
        stub = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [9.0]},
            index=pd.to_datetime(["2026-08-10 01:30"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", return_value=pd.DataFrame()
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=stub), patch.object(
            westock_wrapper, "fetch_yfinance", return_value=pd.DataFrame()
        ):
            result = westock_wrapper.download("0700.HK", period="5d", interval="1h")

        self.assertTrue(result.empty)

    def test_daily_requests_are_unaffected_by_the_intraday_gate(self):
        single_bar = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [9.0]},
            index=pd.to_datetime(["2026-08-10"]),
        )

        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_kline", return_value=single_bar
        ):
            result = westock_wrapper.download("688256.SS", period="18mo", interval="1d")

        self.assertEqual(result.attrs["source"], "tencent")
        self.assertFalse(result.empty)


class TencentRowLimitTests(unittest.TestCase):
    """The limit sent to Tencent was computed in trading days for every interval.

    Verified live: the endpoint returns rows for limit<=2000 and an empty payload
    for limit>=2001 (binary-searched on sh600519 month). A 10y monthly request
    computed 10*250=2500, tripped that cap, and silently fell through to Yahoo —
    which is how a single ticker ended up with qfq daily/weekly and split_only
    monthly in one run.
    """

    def _limit_for(self, interval, period):
        captured = {}

        def fake_tencent(ticker, ws_period, limit):
            captured['limit'] = limit
            captured['ws_period'] = ws_period
            return pd.DataFrame()

        # 1h routes to the minute endpoint and everything else to fqkline/get, so
        # both are stubbed to capture whichever one the interval selects.
        with patch.object(westock_wrapper, "fetch_kline", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_tencent_kline", side_effect=fake_tencent
        ), patch.object(
            westock_wrapper, "fetch_tencent_minute_kline", side_effect=fake_tencent
        ), patch.object(westock_wrapper, "fetch_yahoo_chart", return_value=pd.DataFrame()), patch.object(
            westock_wrapper, "fetch_yfinance", return_value=pd.DataFrame()
        ):
            westock_wrapper.download("688256.SS", period=period, interval=interval)

        return captured

    def test_monthly_limit_is_counted_in_months_not_trading_days(self):
        captured = self._limit_for("1mo", "10y")

        self.assertEqual(captured['ws_period'], 'month')
        self.assertLessEqual(captured['limit'], westock_wrapper.TENCENT_MAX_ROWS)
        # 10 years of monthly bars is ~120, not 2500.
        self.assertGreaterEqual(captured['limit'], 120)
        self.assertLess(captured['limit'], 300)

    def test_weekly_limit_is_counted_in_weeks_not_trading_days(self):
        captured = self._limit_for("1wk", "5y")

        self.assertEqual(captured['ws_period'], 'week')
        # 5 years of weekly bars is ~260, not 1250.
        self.assertGreaterEqual(captured['limit'], 260)
        self.assertLess(captured['limit'], 600)

    def test_daily_limit_still_uses_trading_days(self):
        captured = self._limit_for("1d", "18mo")

        self.assertEqual(captured['ws_period'], 'day')
        self.assertGreaterEqual(captured['limit'], 378)

    def test_every_request_is_clamped_below_the_live_cap(self):
        """limit>=2001 returns an empty payload, so never ask for more."""
        for interval, period in (
            ("1d", "max"),
            ("1d", "20y"),
            ("1wk", "max"),
            ("1mo", "max"),
            ("1h", "5y"),
        ):
            with self.subTest(interval=interval, period=period):
                captured = self._limit_for(interval, period)
                self.assertLessEqual(captured['limit'], westock_wrapper.TENCENT_MAX_ROWS)
                self.assertGreater(captured['limit'], 0)

    def test_cap_matches_the_live_boundary(self):
        self.assertEqual(westock_wrapper.TENCENT_MAX_ROWS, 2000)


class AdjustmentConventionTests(unittest.TestCase):
    """The fallback chain may hand different price bases to different timeframes."""

    def test_every_basis_a_fetcher_can_report_is_a_known_label(self):
        """ADJUSTMENT_LABELS is the set of known bases, so it gates comparability."""
        self.assertEqual(
            set(westock_wrapper.ADJUSTMENT_LABELS),
            {"raw", "split_only", "qfq", "hfq"},
        )
        self.assertNotIn(
            westock_wrapper.ADJUSTMENT_UNKNOWN, westock_wrapper.ADJUSTMENT_LABELS
        )

    def test_same_label_is_comparable(self):
        self.assertTrue(westock_wrapper.adjustments_comparable("qfq", "qfq"))
        self.assertTrue(westock_wrapper.adjustments_comparable("split_only", "split_only"))

    def test_differing_labels_are_not_comparable(self):
        """qfq and hfq both adjust, but anchor the scale at opposite ends."""
        self.assertFalse(westock_wrapper.adjustments_comparable("qfq", "hfq"))
        self.assertFalse(westock_wrapper.adjustments_comparable("qfq", "split_only"))
        self.assertFalse(westock_wrapper.adjustments_comparable("raw", "split_only"))

    def test_unknown_needs_a_matching_source_to_be_comparable(self):
        """Source identity is what resolves an unlabelled basis.

        Two *different* silent providers are not evidence of agreement, but one
        provider cannot disagree with itself — which is the westock-data case,
        since it is first in the chain and supplies every timeframe when present.
        """
        self.assertTrue(
            westock_wrapper.adjustments_comparable("unknown", "unknown", "westock", "westock")
        )
        self.assertFalse(
            westock_wrapper.adjustments_comparable("unknown", "unknown", "westock", "other")
        )
        self.assertFalse(westock_wrapper.adjustments_comparable("unknown", "unknown"))
        self.assertFalse(westock_wrapper.adjustments_comparable("unknown", "qfq"))

    def test_a_known_label_ignores_source_identity(self):
        """qfq from two providers is still qfq; only `unknown` needs the source."""
        self.assertTrue(
            westock_wrapper.adjustments_comparable("qfq", "qfq", "tencent", "westock")
        )
        self.assertFalse(
            westock_wrapper.adjustments_comparable("qfq", "hfq", "tencent", "tencent")
        )

    def test_describe_adjustment_is_human_readable_for_the_report(self):
        self.assertIn("未复权", westock_wrapper.describe_adjustment("raw"))
        self.assertIn("前复权", westock_wrapper.describe_adjustment("qfq"))
        self.assertIn("后复权", westock_wrapper.describe_adjustment("hfq"))
        self.assertIn("拆股", westock_wrapper.describe_adjustment("split_only"))
        self.assertIn("未知", westock_wrapper.describe_adjustment("unknown"))

    def test_describe_adjustment_tolerates_unrecognized_label(self):
        self.assertIn("未知", westock_wrapper.describe_adjustment("something-else"))
