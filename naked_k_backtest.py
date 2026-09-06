"""Offline, causal long/cash backtest for the production trend signal."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import stdev
from typing import Any, Callable

import pandas as pd

from naked_k_config import TradingConfig, load_trading_config
from naked_k_risk import build_risk_plan
from naked_k_trend import MIN_HISTORY, RULE_VERSION, analyze_trend, evaluate_entry, indicator_frame
from naked_k_zones import validate_ohlcv


SignalBuilder = Callable[[pd.DataFrame], dict[str, Any]]
SCHEMA_VERSION = "1.0"


def _date(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _cost(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 10000:
        raise ValueError(f"{name} must be finite and in [0, 10000)")
    return float(value) / 10000


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("daily must be a DataFrame")
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if data.index.hasnans:
        raise ValueError("daily index cannot contain NaT")
    return validate_ohlcv(data)


def build_walk_forward_windows(frame: pd.DataFrame, train_size: int, test_size: int,
                               step: int | None = None) -> list[dict[str, Any]]:
    if type(train_size) is not int or type(test_size) is not int or (step is not None and type(step) is not int):
        raise ValueError("train_size, test_size, and step must be integers")
    if train_size < MIN_HISTORY or test_size <= 0:
        raise ValueError(f"train_size must be at least {MIN_HISTORY} and test_size must be positive")
    stride = test_size if step is None else step
    if stride < test_size:
        raise ValueError("step must be at least test_size so test windows do not overlap")
    windows = []
    for start in range(0, len(frame) - train_size - test_size + 1, stride):
        train, test = frame.iloc[start:start + train_size].copy(), frame.iloc[start + train_size:start + train_size + test_size].copy()
        windows.append({"train": train, "test": test, "train_start": train.index[0], "train_end": train.index[-1],
                        "test_start": test.index[0], "test_end": test.index[-1]})
    return windows


def _daily_returns(equity: list[float]) -> list[float]:
    return [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity)) if equity[i - 1] > 0]


def _exposure_summary(curve: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(point["gross_exposure_pct"]) for point in curve]
    return {"basis": "daily_close_market_value_over_equity",
            "average_gross_exposure_pct": sum(values) / len(values) if values else None,
            "maximum_gross_exposure_pct": max(values) if values else None}


def calculate_performance_metrics(trades: list[dict[str, Any]], equity: list[float]) -> dict[str, Any]:
    returns = _daily_returns(equity)
    start, end = (equity[0], equity[-1]) if equity else (0.0, 0.0)
    peak, max_dd = 0.0, 0.0
    for value in equity:
        peak = max(peak, value)
        if peak:
            max_dd = max(max_dd, (peak - value) / peak)
    wins = [float(t["net_pnl"]) for t in trades if t["net_pnl"] > 0]
    losses = [float(t["net_pnl"]) for t in trades if t["net_pnl"] < 0]
    rs = [float(t["r_multiple"]) for t in trades]
    sharpe = None
    if len(returns) >= 2 and all(math.isfinite(x) for x in returns):
        sigma = stdev(returns)
        if sigma > 0: sharpe = sum(returns) / len(returns) / sigma * math.sqrt(252)
    annualized = None
    if len(equity) > 1 and start > 0 and end >= 0:
        annualized = ((end / start) ** (252 / (len(equity) - 1)) - 1) * 100
    return {"trade_count": len(trades), "total_return_pct": (end / start - 1) * 100 if start else 0.0,
            "annualized_return_pct": annualized, "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe, "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
            "average_r": sum(rs) / len(rs) if rs else 0.0}


def _allocation_cap(config: TradingConfig) -> float:
    p = config.portfolio
    return min(config.risk.action_gross_caps.get("小仓试错", 0), p.max_total_gross_pct,
               p.max_single_name_gross_pct, p.max_direction_gross_pct, p.max_market_gross_pct)


def _comparison(frame: pd.DataFrame, start_pos: int, commission: float, slip: float,
                config: TradingConfig, ema: bool, initial_equity: float) -> dict[str, Any]:
    cap = _allocation_cap(config) / 100
    cash, units, values = initial_equity, 0.0, [initial_equity]
    indicators = indicator_frame(frame) if ema else None
    if ema and (start_pos >= len(indicators) or pd.isna(indicators.iloc[start_pos][["ema50", "ema200"]]).any()):
        return {"name": "ema50_200_long_cash", "status": "not_computable", "allocation_pct": cap * 100,
                "period_start": _date(frame.index[start_pos + 1]) if start_pos + 1 < len(frame) else None,
                "period_end": _date(frame.index[-1]), "cost_adjusted": True, "metrics": None, "equity": [],
                "terminal_position_open": False, "boundary_liquidation_fee_applied": False,
                "reason": "EMA200 unavailable at the initial evaluation signal"}
    for i in range(start_pos + 1, len(frame)):
        row = frame.iloc[i]
        prior = indicators.iloc[i - 1] if indicators is not None else None
        enter = units == 0 and (not ema or (prior.Close > prior.ema50 > prior.ema200))
        exit_now = units and ema and prior.Close < prior.ema50
        if exit_now:
            fill = row.Open * (1 - slip)
            cash += units * fill * (1 - commission)
            units = 0
        if enter:
            fill = row.Open * (1 + slip)
            current_equity = cash
            units = min(current_equity * cap / fill, cash / (fill * (1 + commission)))
            cash -= units * fill * (1 + commission)
        values.append(cash + units * row.Close)
    label = "ema50_200_long_cash" if ema else "buy_hold"
    metrics = calculate_performance_metrics([], values)
    return {"name": label, "status": "completed", "allocation_pct": cap * 100, "period_start": _date(frame.index[start_pos + 1]) if start_pos + 1 < len(frame) else None,
            "period_end": _date(frame.index[-1]), "cost_adjusted": True, "metrics": metrics,
            "equity": values, "terminal_position_open": bool(units),
            "boundary_liquidation_fee_applied": False}


def _metadata(frame: pd.DataFrame, config: TradingConfig, commission_bps: float,
              slippage_bps: float, initial_equity: float, evaluation_start: str | None,
              minimum_history: int) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION,
            "commission_bps": float(commission_bps), "slippage_bps": float(slippage_bps),
            "config": asdict(config), "initial_equity": float(initial_equity), "minimum_history": minimum_history,
            "sample_start": _date(frame.index[0]) if len(frame) else None,
            "sample_end": _date(frame.index[-1]) if len(frame) else None,
            "evaluation_start": evaluation_start, "evaluation_end": _date(frame.index[-1]) if evaluation_start else None,
            "annualization": {"periods_per_year": 252, "risk_free_rate": 0.0},
            "validation_status": "UNVALIDATED"}


def _actionable(signal: dict[str, Any]) -> bool:
    return (signal.get("entry_candidate") is True and signal.get("direction") == "up"
            and signal.get("action", "小仓试错") in {"买入", "小仓试错"})


def run_event_backtest(name: str, ticker: str, daily: pd.DataFrame, *, commission_bps: float,
                       slippage_bps: float, config: TradingConfig | None = None, min_history: int = MIN_HISTORY,
                       signal_builder: SignalBuilder | None = None, initial_equity: float = 100000.0) -> dict[str, Any]:
    commission, slip = _cost(commission_bps, "commission_bps"), _cost(slippage_bps, "slippage_bps")
    if type(min_history) is not int or min_history < 2 or isinstance(initial_equity, bool) or not isinstance(initial_equity, (int, float)) or initial_equity <= 0 or not math.isfinite(initial_equity):
        raise ValueError("min_history must be at least 2 and initial_equity must be finite and positive")
    frame, config, builder = _clean(daily), config or TradingConfig(), signal_builder or analyze_trend
    if signal_builder is None and min_history < MIN_HISTORY:
        raise ValueError(f"the production trend strategy requires min_history >= {MIN_HISTORY}")
    evaluation_start = _date(frame.index[min_history]) if len(frame) > min_history else None
    metadata = _metadata(frame, config, commission_bps, slippage_bps, initial_equity, evaluation_start, min_history)
    if len(frame) <= min_history:
        return {"name": name, "ticker": ticker, "status": "insufficient_history", "trades": [], "open_position": None,
                "equity_curve": [], "metrics": None, "comparisons": {}, "metadata": metadata,
                "exposure": _exposure_summary([]),
                "external_benchmark": {"return_pct": None, "excess_return_pct": None},
                "audit": {"no_lookahead": True, "entries": 0, "rejected_entries": [], "cost_label": "gross research" if not commission and not slip else "cost adjusted"}}

    cash, position, pending_entry, pending_exit = initial_equity, None, None, False
    trades, rejected, curve, mode_transitions = [], [], [], []
    peak, consecutive_losses = initial_equity, 0
    first = min_history - 1
    curve.append({"date": _date(frame.index[first]), "equity": initial_equity, "daily_return": None,
                  "gross_exposure_pct": 0.0})
    signal = builder(frame.iloc[:first + 1].copy())
    if signal.get("history_mode"):
        mode_transitions.append({key: signal.get(key) for key in
            ("daily_rows", "history_mode", "direction_basis", "rule_version")})
    if _actionable(signal):
        pending_entry = signal

    for i in range(first + 1, len(frame)):
        row, day = frame.iloc[i], frame.index[i]
        stopped_today = False
        if position:
            if row.Open <= position["stop"]:
                base, reason = row.Open, "stop_gap"
            elif pending_exit:
                base, reason = row.Open, "trend_exit"
            elif row.Low <= position["stop"]:
                base, reason = position["stop"], "stop"
            else:
                base = reason = None
            if base is not None:
                fill = float(base) * (1 - slip)
                exit_commission = position["units"] * fill * commission
                cash += position["units"] * fill - exit_commission
                pnl = cash - position["cash_before_entry"]
                risk = position["initial_dollar_risk"]
                trade = {**{k: position[k] for k in ("entry_date", "entry_price", "units")}, "exit_date": _date(day),
                         "exit_price": fill, "direction": "long", "exit_reason": reason,
                         "holding_days": i - position["entry_index"], "net_pnl": pnl,
                         "r_multiple": pnl / risk if risk > 0 else None}
                trades.append(trade)
                consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                position, pending_exit, stopped_today = None, False, True

        if position is None and pending_entry is not None and not stopped_today:
            fill = float(row.Open) * (1 + slip)
            check = evaluate_entry(pending_entry, fill)
            if check["eligible"] and _actionable(pending_entry):
                current_equity = cash
                drawdown = max(0.0, (peak - current_equity) / peak * 100) if peak else 0.0
                plan = build_risk_plan("小仓试错", fill, float(check["stop"]), None,
                    current_drawdown_pct=drawdown, consecutive_losses=consecutive_losses, config=config.risk)
                account_risk_gross = config.portfolio.max_total_account_risk_pct / plan["risk_pct"] * 100
                gross = min(plan["suggested_gross_pct"], _allocation_cap(config), account_risk_gross) / 100
                units = min(current_equity * gross / fill, cash / (fill * (1 + commission))) if gross > 0 else 0
                if units > 0:
                    before, entry_commission = cash, units * fill * commission
                    cash -= units * fill + entry_commission
                    position = {"entry_date": _date(day), "entry_price": fill, "entry_index": i, "units": units,
                                "stop": float(check["stop"]), "initial_dollar_risk": units * (fill - float(check["stop"])),
                                "cash_before_entry": before}
                    if row.Low <= position["stop"]:
                        exit_fill = position["stop"] * (1 - slip)
                        exit_commission = units * exit_fill * commission
                        cash += units * exit_fill - exit_commission
                        pnl = cash - before
                        trades.append({"entry_date": _date(day), "entry_price": fill, "exit_date": _date(day),
                            "exit_price": exit_fill, "direction": "long", "exit_reason": "stop", "holding_days": 0,
                            "units": units, "net_pnl": pnl, "r_multiple": pnl / position["initial_dollar_risk"]})
                        consecutive_losses = consecutive_losses + 1 if pnl < 0 else 0
                        position = None
            else:
                rejected.append({"signal_date": pending_entry.get("as_of"), "execution_date": _date(day), "reason": check["reason"]})
            pending_entry = None

        equity = cash + (position["units"] * row.Close if position else 0)
        previous = curve[-1]["equity"]
        exposure = position["units"] * row.Close / equity * 100 if position and equity > 0 else 0.0
        curve.append({"date": _date(day), "equity": equity, "daily_return": equity / previous - 1,
                      "gross_exposure_pct": exposure})
        peak = max(peak, equity)
        signal = builder(frame.iloc[:i + 1].copy())
        if signal.get("history_mode") and (not mode_transitions or signal.get("history_mode") != mode_transitions[-1]["history_mode"]):
            mode_transitions.append({key: signal.get(key) for key in
                ("daily_rows", "history_mode", "direction_basis", "rule_version")})
        if position:
            new_stop = signal.get("stop_loss")
            if isinstance(new_stop, (int, float)) and math.isfinite(new_stop) and position["stop"] < new_stop < row.Close:
                position["stop"] = float(new_stop)
            pending_exit = bool(signal.get("exit_signal"))
        elif _actionable(signal):
            pending_entry = signal

    equity_values = [point["equity"] for point in curve]
    comparisons = {"buy_hold": _comparison(frame, first, commission, slip, config, False, initial_equity),
                   "ema50_200": _comparison(frame, first, commission, slip, config, True, initial_equity)}
    public_position = None if not position else {k: position[k] for k in ("entry_date", "entry_price", "units", "stop", "initial_dollar_risk")}
    return {"name": name, "ticker": ticker, "status": "completed", "trades": trades, "open_position": public_position,
            "equity_curve": curve, "metrics": calculate_performance_metrics(trades, equity_values), "comparisons": comparisons,
            "metadata": metadata, "exposure": _exposure_summary(curve),
            "external_benchmark": {"return_pct": None, "excess_return_pct": None},
            "audit": {"no_lookahead": True, "entries": len(trades) + bool(position), "rejected_entries": rejected,
                      "evaluated_from": _date(frame.index[first]), "cost_label": "gross research" if not commission and not slip else "cost adjusted",
                      "fractional_units": True, "signal_mode_transitions": mode_transitions}}


def run_walk_forward_event_backtest(name: str, ticker: str, daily: pd.DataFrame, train_size: int,
                                    test_size: int, *, commission_bps: float, slippage_bps: float,
                                    step: int | None = None, config: TradingConfig | None = None,
                                    signal_builder: SignalBuilder | None = None) -> dict[str, Any]:
    _cost(commission_bps, "commission_bps")
    _cost(slippage_bps, "slippage_bps")
    frame = _clean(daily)
    windows = build_walk_forward_windows(frame, train_size, test_size, step)
    active_config = config or TradingConfig()
    results, trades, stitched, equity = [], [], [], 100000.0
    for number, window in enumerate(windows, 1):
        combined = pd.concat([window["train"], window["test"]])
        result = run_event_backtest(name, ticker, combined, commission_bps=commission_bps,
            slippage_bps=slippage_bps, config=active_config, min_history=train_size,
            signal_builder=signal_builder, initial_equity=100000.0)
        test_curve = result["equity_curve"][1:]
        for point in test_curve:
            equity *= 1 + point["daily_return"]
            stitched.append({"date": point["date"], "equity": equity, "daily_return": point["daily_return"],
                             "gross_exposure_pct": point["gross_exposure_pct"]})
        tagged = [{**trade, "window": number} for trade in result["trades"]]
        trades.extend(tagged)
        results.append({"window": number, "train_start": _date(window["train_start"]), "train_end": _date(window["train_end"]),
                        "test_start": _date(window["test_start"]), "test_end": _date(window["test_end"]),
                        "trades": tagged, "open_position": result["open_position"],
                        "starting_equity": result["equity_curve"][0]["equity"],
                        "ending_equity": result["equity_curve"][-1]["equity"],
                        "equity_curve": result["equity_curve"], "metrics": result["metrics"],
                        "comparisons": result["comparisons"], "audit": result["audit"]})
    values = [100000.0] + [p["equity"] for p in stitched]
    comparisons = {}
    for key in ("buy_hold", "ema50_200"):
        aggregate = [100000.0]
        unavailable = next((result["comparisons"][key] for result in results
                            if result["comparisons"][key]["status"] != "completed"), None)
        if unavailable:
            comparisons[key] = {**unavailable, "period_start": _date(windows[0]["test_start"]),
                "period_end": _date(windows[-1]["test_end"]), "reset_policy": "reset_each_test_window"}
            continue
        for result in results:
            window_values = result["comparisons"][key]["equity"]
            for daily_return in _daily_returns(window_values):
                aggregate.append(aggregate[-1] * (1 + daily_return))
        if windows:
            sample = results[0]["comparisons"][key]
            comparisons[key] = {"name": sample["name"], "status": "completed", "allocation_pct": sample["allocation_pct"],
                "period_start": _date(windows[0]["test_start"]), "period_end": _date(windows[-1]["test_end"]),
                "cost_adjusted": True, "metrics": calculate_performance_metrics([], aggregate), "equity": aggregate,
                "terminal_position_open": any(result["comparisons"][key]["terminal_position_open"] for result in results),
                "reset_policy": "reset_each_test_window", "boundary_liquidation_fee_applied": False}
    evaluation_start = _date(windows[0]["test_start"]) if windows else None
    metadata = _metadata(frame, active_config, commission_bps, slippage_bps, 100000.0, evaluation_start, train_size)
    metadata["evaluation_end"] = _date(windows[-1]["test_end"]) if windows else None
    return {"status": "completed" if windows else "insufficient_history", "windows": results,
            "trades": trades, "equity_curve": stitched,
            "metrics": calculate_performance_metrics(trades, values) if windows else None,
            "comparisons": comparisons, "metadata": metadata, "exposure": _exposure_summary(stitched),
            "external_benchmark": {"return_pct": None, "excess_return_pct": None},
            "audit": {"no_lookahead": True, "mode": "stitched_independent_windows", "window_count": len(results),
                      "training_optimization": False, "return_synthesis": "independent_window_marked_returns",
                      "continuous_strategy_equity": False, "window_state_reset": True,
                      "boundary_liquidation_fee_applied": False}}


def _read_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return _clean(data)


def _read_benchmark(path: str) -> pd.Series:
    data = pd.read_csv(path, parse_dates=["Date"])
    if "Close" not in data or data.Date.isna().any() or data.Date.duplicated().any():
        raise ValueError("benchmark CSV requires unique Date and Close columns")
    values = pd.to_numeric(data.Close, errors="raise")
    if not values.map(lambda value: math.isfinite(value) and value > 0).all():
        raise ValueError("benchmark Close must be finite and positive")
    return pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(data.Date)).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline naked-K trend backtest")
    parser.add_argument("ticker"); parser.add_argument("--csv", required=True)
    parser.add_argument("--commission-bps", required=True, type=float); parser.add_argument("--slippage-bps", required=True, type=float)
    parser.add_argument("--output", required=True); parser.add_argument("--benchmark-csv"); parser.add_argument("--config")
    args = parser.parse_args()
    result = run_event_backtest(args.ticker, args.ticker, _read_csv(args.csv), commission_bps=args.commission_bps,
                                slippage_bps=args.slippage_bps, config=load_trading_config(args.config))
    if args.benchmark_csv:
        benchmark = _read_benchmark(args.benchmark_csv).reindex(pd.to_datetime([p["date"] for p in result["equity_curve"]]))
        if benchmark.notna().all() and len(benchmark) > 1:
            benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1
            result["external_benchmark"] = {"return_pct": benchmark_return * 100,
                "excess_return_pct": result["metrics"]["total_return_pct"] - benchmark_return * 100}
        else:
            result["external_benchmark"] = {"return_pct": None, "excess_return_pct": None}
    output = Path(args.output).resolve()
    reports = (Path(__file__).parent / "reports").resolve()
    if not output.is_relative_to(reports):
        raise ValueError("--output must be under the repository reports/ directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                                 default=lambda value: value.item()), encoding="utf-8")


if __name__ == "__main__":
    main()
