from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from naked_k_zones import detect_price_zones, validate_ohlcv

MIN_HISTORY = 55
FULL_HISTORY = 205
RULE_VERSION = 'technical-trend-v2'


def _validated(frame: pd.DataFrame) -> pd.DataFrame:
    return validate_ohlcv(frame)


def _smooth(values: pd.Series, period: int, alpha: float) -> pd.Series:
    source = values.to_numpy(dtype=float)
    output = np.full(len(source), np.nan)
    if len(values) >= period:
        output[period - 1] = source[:period].mean()
        for i in range(period, len(source)):
            output[i] = output[i - 1] + alpha * (source[i] - output[i - 1])
    return pd.Series(output, index=values.index)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame.Close.shift()
    return pd.concat([frame.High - frame.Low, (frame.High - previous).abs(), (frame.Low - previous).abs()], axis=1).max(axis=1)


def _adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    up, down = frame.High.diff(), -frame.Low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    atr = _smooth(_true_range(frame), period, 1 / period)
    plus_di = (100 * _smooth(plus, period, 1 / period) / atr).where(atr != 0, 0.0)
    minus_di = (100 * _smooth(minus, period, 1 / period) / atr).where(atr != 0, 0.0)
    denominator = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / denominator).where(denominator != 0, 0.0)
    adx = _smooth(dx.iloc[period - 1:], period, 1 / period)
    return adx.reindex(frame.index)


def indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = _validated(frame)
    for period in (20, 50, 200):
        result[f"ema{period}"] = _smooth(result.Close, period, 2 / (period + 1))
    result["atr"] = _smooth(_true_range(result), 14, 1 / 14)
    result["atr_pct"] = result.atr / result.Close * 100
    result["adx"] = _adx(result)
    result["channel_high"] = result.High.shift().rolling(20).max()
    result["channel_low"] = result.Low.shift().rolling(20).min()
    baseline = result.Volume.shift().rolling(20).mean()
    result["relative_volume"] = (result.Volume / baseline).where((result.Volume > 0) & (baseline > 0))
    result["ema50_slope"] = result.ema50 - result.ema50.shift(5)
    return result


def _number(value: Any) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def analyze_trend(frame: pd.DataFrame) -> dict[str, Any]:
    data = indicator_frame(frame)
    if data.empty:
        raise ValueError("OHLCV must contain at least one row")
    row, ready = data.iloc[-1], len(data) >= MIN_HISTORY
    mode = 'full' if len(data) >= FULL_HISTORY else 'short' if ready else 'insufficient'
    close, ema50, ema200, slope = map(_number, [row.Close, row.ema50, row.ema200, row.ema50_slope])
    direction = "unknown"
    if ready:
        fast, slow = (_number(row.ema20), ema50) if mode == 'short' else (ema50, ema200)
        direction = "up" if close > fast > slow and slope > 0 else "down" if close < fast < slow and slope < 0 else "transition"
    adx = _number(row.adx)
    strength = "unknown" if adx is None else "strong" if adx >= 25 else "developing" if adx >= 20 else "weak"
    prior_high = _number(row.channel_high)
    breakout = bool(ready and prior_high is not None and close > prior_high)
    if breakout:
        previous = data.iloc[-2]
        breakout = bool(_number(previous.channel_high) is not None and previous.Close <= previous.channel_high)
    zones = detect_price_zones(frame, close=close)
    atr, prior_low, ema20 = map(_number, [row.atr, row.channel_low, row.ema20])
    stop = prior_low - .5 * atr if prior_low is not None and atr is not None else None
    stop = stop if stop is not None and 0 < stop < close else None
    max_entry = min(close + .5 * atr, ema20 + 2 * atr) if atr is not None and ema20 is not None else None
    resistance = _number(zones["nearest_resistance"]["lower"]) if zones["nearest_resistance"] else None
    enough_room = resistance is None or (stop is not None and resistance - close >= close - stop)
    candidate = bool(ready and direction == "up" and breakout and stop is not None and max_entry is not None
                     and close <= max_entry and enough_room)
    reasons = [f"trend_{direction}", "fresh_breakout" if breakout else "no_fresh_breakout"]
    if resistance is not None and not enough_room:
        reasons.append("resistance_within_1r")
    keys = ["ema20", "ema50", "ema200", "atr", "atr_pct", "adx", "channel_high", "channel_low", "relative_volume", "ema50_slope"]
    return {"status": "ready" if ready else "insufficient_history", "direction": direction, "strength": strength,
            "history_mode": mode, "daily_rows": len(data), "rule_version": RULE_VERSION,
            "direction_basis": {'full': 'ema50_200', 'short': 'ema20_50', 'insufficient': 'unavailable'}[mode],
            "as_of": pd.Timestamp(data.index[-1]).isoformat(), "close": close,
            "indicators": {key: _number(row[key]) for key in keys}, "zones": zones, "breakout": breakout,
            "entry_candidate": candidate, "exit_signal": bool(ready and (close < ema50 or direction == "down")),
            "entry_reference": close, "stop_loss": stop, "max_entry": max_entry, "resistance": resistance,
            "reasons": reasons, "validation_status": "UNVALIDATED"}


def evaluate_entry(signal: dict[str, Any], open_price: float) -> dict[str, Any]:
    entry, stop, resistance = _number(open_price), _number(signal.get("stop_loss")), _number(signal.get("resistance"))
    result = {"eligible": False, "reason": "not_a_candidate", "entry": entry, "stop": stop,
              "risk_per_share": None, "resistance": resistance, "reward_to_risk": None}
    if entry is None or entry <= 0:
        result["reason"] = "invalid_open"
    elif not signal.get("entry_candidate"):
        pass
    elif entry > signal.get("max_entry", -np.inf):
        result["reason"] = "open_above_max_entry"
    elif stop is None or entry <= stop:
        result["reason"] = "invalid_stop"
    else:
        risk = entry - stop
        reward = resistance - entry if resistance is not None else None
        ratio = reward / risk if reward is not None else None
        result.update(risk_per_share=risk, reward_to_risk=ratio)
        if resistance is not None and (reward <= 0 or ratio < 1):
            result["reason"] = "insufficient_resistance_room"
        else:
            result.update(eligible=True, reason="eligible")
    return result
