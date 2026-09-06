import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import naked_k_backtest as bt


def bars(opens, lows=None, closes=None):
    opens = list(map(float, opens)); lows = list(map(float, lows or opens)); closes = list(map(float, closes or opens))
    return pd.DataFrame({"Open": opens, "High": [max(o, c) + 2 for o, c in zip(opens, closes)],
        "Low": lows, "Close": closes, "Volume": [1000.0] * len(opens)},
        index=pd.date_range("2025-01-01", periods=len(opens), freq="D"))


def scripted(signals):
    calls = []
    def build(frame):
        calls.append((len(frame), frame.index[-1]))
        return signals.get(len(frame), {"direction": "transition", "entry_candidate": False,
            "exit_signal": False, "stop_loss": None, "max_entry": None,
            "entry_reference": float(frame.Close.iloc[-1]), "resistance": None, "indicators": {}})
    build.calls = calls
    return build


def candidate(stop=90.0, maximum=101.5, action="小仓试错"):
    return {"direction": "up", "entry_candidate": True, "exit_signal": False,
        "stop_loss": stop, "max_entry": maximum, "entry_reference": 100.0,
        "resistance": None, "indicators": {}, "action": action}


class BacktestTests(unittest.TestCase):
    def test_gap_entry_is_rejected_at_actual_slipped_open(self):
        result = bt.run_event_backtest("x", "X", bars([100, 100, 120]), commission_bps=0,
            slippage_bps=0, min_history=2, signal_builder=scripted({2: candidate()}))
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["audit"]["rejected_entries"][0]["reason"], "open_above_max_entry")

    def test_gap_stop_uses_worse_open_and_costs(self):
        frame = bars([100, 100, 100, 80], lows=[99, 99, 95, 79])
        result = bt.run_event_backtest("x", "X", frame, commission_bps=10,
            slippage_bps=100, min_history=2, signal_builder=scripted({2: candidate(stop=90, maximum=110)}))
        trade = result["trades"][0]
        self.assertAlmostEqual(trade["entry_price"], 101.0)
        self.assertAlmostEqual(trade["exit_price"], 79.2)
        units = trade["units"]
        self.assertAlmostEqual(trade["net_pnl"], units * (79.2 - 101) - units * (101 + 79.2) * .001)
        self.assertEqual((trade["holding_days"], trade["exit_reason"]), (1, "stop_gap"))

    def test_multiday_position_no_duplicates_and_remains_open_at_end(self):
        frame = bars([100, 100, 100, 103, 106], lows=[99, 99, 95, 96, 99])
        signals = {i: candidate(stop=88 + i, maximum=110) for i in range(2, 6)}
        result = bt.run_event_backtest("x", "X", frame, commission_bps=0,
            slippage_bps=0, min_history=2, signal_builder=scripted(signals))
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["open_position"]["entry_date"], "2025-01-03")
        self.assertEqual(result["open_position"]["stop"], 93.0)
        self.assertEqual(result["audit"]["entries"], 1)
        self.assertGreater(result["metrics"]["total_return_pct"], 0)
        self.assertEqual(result["equity_curve"][0]["gross_exposure_pct"], 0)
        self.assertGreater(result["equity_curve"][1]["gross_exposure_pct"], 0)
        self.assertGreater(result["exposure"]["average_gross_exposure_pct"], 0)
        self.assertGreaterEqual(result["exposure"]["maximum_gross_exposure_pct"],
                                result["exposure"]["average_gross_exposure_pct"])

    def test_trend_exit_next_open_and_future_append_is_causal(self):
        base = bars([100, 100, 100, 105], lows=[99, 99, 95, 100])
        signals = {2: candidate(stop=90, maximum=110), 3: {**candidate(), "entry_candidate": False, "exit_signal": True}}
        first = bt.run_event_backtest("x", "X", base, commission_bps=0, slippage_bps=0,
            min_history=2, signal_builder=scripted(signals))
        extra = bars([999]); extra.index = [base.index[-1] + pd.Timedelta(days=1)]
        builder = scripted(signals)
        extended = bt.run_event_backtest("x", "X", pd.concat([base, extra]), commission_bps=0,
            slippage_bps=0, min_history=2, signal_builder=builder)
        self.assertEqual(first["trades"][0], extended["trades"][0])
        self.assertEqual(first["trades"][0]["exit_date"], "2025-01-04")
        self.assertTrue(all(length == i for i, (length, _) in enumerate(builder.calls, start=2)))

    def test_daily_sharpe_uses_equity_returns(self):
        metrics = bt.calculate_performance_metrics([], [100, 110, 99])
        returns = [.1, -.1]
        expected = (sum(returns) / 2) / math.sqrt(sum((x - sum(returns) / 2) ** 2 for x in returns)) * math.sqrt(252)
        self.assertAlmostEqual(metrics["sharpe_ratio"], expected)

    def test_ema_comparison_resizes_from_current_equity(self):
        frame = bars([100, 100, 200, 100], closes=[100, 200, 200, 200])
        indicators = frame.copy()
        indicators["ema50"] = [90, 210, 90, 90]
        indicators["ema200"] = [80, 100, 80, 80]
        with patch.object(bt, "indicator_frame", return_value=indicators):
            result = bt._comparison(frame, 0, 0, 0, bt.TradingConfig(), True, 100)
        self.assertAlmostEqual(result["equity"][-1], 132.25)

    def test_avoid_action_cannot_enter(self):
        for action in ("回避", "减仓", "unexpected"):
            result = bt.run_event_backtest("x", "X", bars([100, 100, 100]), commission_bps=0,
                slippage_bps=0, min_history=2, signal_builder=scripted({2: candidate(action=action)}))
            self.assertEqual(result["audit"]["entries"], 0)
        malformed = {**candidate(), "direction": "down"}
        result = bt.run_event_backtest("x", "X", bars([100, 100, 100]), commission_bps=0,
            slippage_bps=0, min_history=2, signal_builder=scripted({2: malformed}))
        self.assertEqual(result["audit"]["entries"], 0)

    def test_same_day_entry_stop_and_trailing_stop_only_next_bar(self):
        same_day = bars([100, 100, 100], lows=[99, 99, 89])
        result = bt.run_event_backtest("x", "X", same_day, commission_bps=0, slippage_bps=0,
            min_history=2, signal_builder=scripted({2: candidate(stop=90, maximum=110)}))
        self.assertEqual((result["trades"][0]["holding_days"], result["trades"][0]["exit_price"]), (0, 90))
        self.assertEqual(result["equity_curve"][-1]["gross_exposure_pct"], 0)

        frame = bars([100, 100, 100, 96], lows=[99, 99, 94, 94])
        signals = {2: candidate(stop=90, maximum=110), 3: {**candidate(stop=95, maximum=110), "entry_candidate": False}}
        result = bt.run_event_backtest("x", "X", frame, commission_bps=0, slippage_bps=0,
            min_history=2, signal_builder=scripted(signals))
        self.assertEqual(result["trades"][0]["exit_date"], "2025-01-04")
        self.assertEqual(result["trades"][0]["exit_price"], 95)

    def test_missing_volume_values_follow_shared_validation_and_account_risk_cap(self):
        frame = bars([100, 100, 100]); frame.loc[frame.index[0], "Volume"] = float("nan")
        from naked_k_config import PortfolioConfig, TradingConfig
        config = TradingConfig(portfolio=PortfolioConfig(max_total_account_risk_pct=.2))
        result = bt.run_event_backtest("x", "X", frame, commission_bps=0, slippage_bps=0,
            config=config, min_history=2, signal_builder=scripted({2: candidate(stop=50, maximum=110)}))
        position = result["open_position"]
        self.assertLessEqual(position["initial_dollar_risk"], 200 + 1e-9)
        invalid = frame.copy(); invalid.index = [pd.NaT, *frame.index[1:]]
        with self.assertRaises(ValueError):
            bt.run_event_backtest("x", "X", invalid, commission_bps=0, slippage_bps=0,
                min_history=2, signal_builder=scripted({}))

    def test_metadata_and_insufficient_history_are_explicit(self):
        result = bt.run_event_backtest("x", "X", bars([100, 101]), commission_bps=1, slippage_bps=2)
        self.assertEqual(result["status"], "insufficient_history")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["external_benchmark"], {"return_pct": None, "excess_return_pct": None})
        metadata = result["metadata"]
        self.assertEqual((metadata["commission_bps"], metadata["slippage_bps"]), (1.0, 2.0))
        self.assertEqual(metadata["validation_status"], "UNVALIDATED")
        self.assertEqual(metadata["annualization"], {"periods_per_year": 252, "risk_free_rate": 0.0})
        exactly_warm = bt.run_event_backtest("x", "X", bars([100, 101]), commission_bps=0,
            slippage_bps=0, min_history=2, signal_builder=scripted({}))
        self.assertEqual(exactly_warm["status"], "insufficient_history")

    def test_walk_forward_windows_do_not_overlap_and_stitch_unique_sessions(self):
        frame = bars(range(100, 320))
        with self.assertRaises(ValueError): bt.build_walk_forward_windows(frame, 205, 5, step=4)
        windows = bt.build_walk_forward_windows(frame, 205, 5)
        self.assertEqual(len(windows), 3)
        result = bt.run_walk_forward_event_backtest("x", "X", frame, 205, 5,
            commission_bps=0, slippage_bps=0, signal_builder=scripted({}))
        dates = [point["date"] for point in result["equity_curve"]]
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(result["comparisons"]["buy_hold"]["period_start"], windows[0]["test_start"].strftime("%Y-%m-%d"))
        self.assertEqual(result["audit"]["mode"], "stitched_independent_windows")

    def test_walk_forward_preserves_open_window_positions_and_explains_marked_synthesis(self):
        frame = bars([100] * 205 + list(range(100, 110)), lows=[99] * 215)
        result = bt.run_walk_forward_event_backtest("x", "X", frame, 205, 5,
            commission_bps=0, slippage_bps=0,
            signal_builder=scripted({205: candidate(stop=90, maximum=110)}))
        self.assertEqual(result["trades"], [])
        self.assertNotEqual(result["metrics"]["total_return_pct"], 0)
        for window in result["windows"]:
            self.assertIsNotNone(window["open_position"])
            self.assertTrue(window["equity_curve"])
            self.assertIn("audit", window)
            self.assertEqual(window["ending_equity"], window["equity_curve"][-1]["equity"])
        self.assertEqual(result["audit"]["return_synthesis"], "independent_window_marked_returns")
        self.assertFalse(result["audit"]["boundary_liquidation_fee_applied"])
        for comparison in result["comparisons"].values():
            self.assertEqual(comparison["reset_policy"], "reset_each_test_window")
            self.assertIn("terminal_position_open", comparison)

    def test_cost_arguments_are_required_and_validated(self):
        frame = bars([1, 1, 1])
        with self.assertRaises(TypeError): bt.run_event_backtest("x", "X", frame)
        with self.assertRaises(ValueError):
            bt.run_event_backtest("x", "X", frame, commission_bps=0, slippage_bps=0,
                                  min_history=2.5, signal_builder=scripted({}))
        for bad in (-1, float("nan"), 10000):
            with self.assertRaises(ValueError):
                bt.run_event_backtest("x", "X", frame, commission_bps=bad, slippage_bps=0, min_history=2)
        with self.assertRaises(ValueError): bt.build_walk_forward_windows(frame, 205.5, 1)
        with self.assertRaises(ValueError):
            bt.run_walk_forward_event_backtest("x", "X", bars(range(300)), 205, 5,
                commission_bps=float("nan"), slippage_bps=0)

    def test_cli_smoke_and_benchmark_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "prices.csv"
            bars(range(100, 305)).rename_axis("Date").to_csv(source)
            output = Path(bt.__file__).parent / "reports" / "nested" / "result.json"
            argv = ["backtest", "X", "--csv", str(source), "--commission-bps", "0",
                    "--slippage-bps", "0", "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(Path, "mkdir"), patch.object(Path, "write_text", return_value=1) as write:
                bt.main()
                json.loads(write.call_args.args[0])
            with patch.object(sys, "argv", [*argv[:-1], str(Path(directory) / "outside.json")]):
                with self.assertRaises(ValueError): bt.main()
            bad_benchmark = Path(directory) / "benchmark.csv"
            pd.DataFrame({"Date": ["2025-01-01"], "Close": [-1]}).to_csv(bad_benchmark, index=False)
            with self.assertRaises(ValueError): bt._read_benchmark(str(bad_benchmark))
            pd.DataFrame({"Date": ["2025-01-01"], "Price": [1]}).to_csv(bad_benchmark, index=False)
            with self.assertRaises(ValueError): bt._read_benchmark(str(bad_benchmark))
            pd.DataFrame({"Date": ["not-a-date"], "Close": [1]}).to_csv(bad_benchmark, index=False)
            with self.assertRaises(ValueError): bt._read_benchmark(str(bad_benchmark))


if __name__ == "__main__": unittest.main()
