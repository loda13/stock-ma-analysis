from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _date(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.date().isoformat() if timestamp == timestamp.normalize() else timestamp.isoformat()


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("OHLCV must have a sorted, unique DatetimeIndex")
    if any(column not in frame for column in columns):
        raise ValueError("OHLCV columns Open, High, Low, Close, Volume are required")
    clean = frame.copy()
    clean[columns] = clean[columns].astype(float)
    prices = clean[columns[:4]]
    if (not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
            or (clean.High < prices[["Open", "Low", "Close"]].max(axis=1)).any()
            or (clean.Low > prices[["Open", "High", "Close"]].min(axis=1)).any()):
        raise ValueError("OHLCV price data must be finite, positive, and have valid high/low geometry")
    if ((clean.Volume.notna()) & (~np.isfinite(clean.Volume) | (clean.Volume < 0))).any():
        raise ValueError("OHLCV Volume must be finite and non-negative when present")
    return clean


def _swings(clean: pd.DataFrame, window: int) -> list[dict[str, Any]]:
    if window < 1:
        raise ValueError("swing_window must be positive")
    points = []
    for position in range(window, len(clean) - window):
        row = clean.iloc[position]
        neighbors = clean.iloc[position - window:position + window + 1].drop(clean.index[position])
        common = {"confirmed_at": _date(clean.index[position + window]),
                  "swing_date": _date(clean.index[position]), "source": "structural_swing", "_position": position}
        if row.High > neighbors.High.max():
            lower, upper = max(row.Open, row.Close), row.High
            points.append({"kind": "resistance", "zone_type": "supply", "lower": float(lower), "upper": float(upper),
                           "midpoint": float((lower + upper) / 2), **common})
        if row.Low < neighbors.Low.min():
            lower, upper = row.Low, min(row.Open, row.Close)
            points.append({"kind": "support", "zone_type": "demand", "lower": float(lower), "upper": float(upper),
                           "midpoint": float((lower + upper) / 2), **common})
    return points


def _avwap(clean: pd.DataFrame, points: list[dict[str, Any]], close: float) -> dict[str, Any] | None:
    kind = "support" if close >= clean.Close.iloc[0] else "resistance"
    candidates = [point for point in points if point["kind"] == kind]
    if not candidates:
        return None
    anchor = candidates[-1]
    anchored = clean.iloc[anchor["_position"]:]
    if anchored.Volume.isna().any():
        return None
    total = float(anchored.Volume.sum(skipna=True))
    if total <= 0:
        return None
    typical = (anchored.High + anchored.Low + anchored.Close) / 3
    return {"anchor_date": anchor["swing_date"], "confirmed_at": anchor["confirmed_at"],
            "anchor_type": "swing_low" if kind == "support" else "swing_high",
            "value": float((typical * anchored.Volume).sum(skipna=True) / total), "source": "structural_swing"}


def detect_price_zones(frame: pd.DataFrame, close: float | None = None, swing_window: int = 2) -> dict[str, Any]:
    clean = validate_ohlcv(frame)
    if clean.empty:
        return {"zones": [], "nearest_support": None, "nearest_resistance": None, "anchored_vwap": None}
    latest = float(clean.Close.iloc[-1] if close is None else close)
    if not np.isfinite(latest) or latest <= 0:
        raise ValueError("close must be finite and positive")
    internal = _swings(clean, swing_window)
    zones = [{key: value for key, value in point.items() if not key.startswith("_")} for point in internal]
    supports = [zone for zone in zones if zone["kind"] == "support" and zone["midpoint"] < latest]
    resistances = [zone for zone in zones if zone["kind"] == "resistance" and zone["upper"] >= latest]
    return {"zones": zones,
            "nearest_support": max(supports, key=lambda zone: zone["midpoint"], default=None),
            "nearest_resistance": min(resistances, key=lambda zone: zone["lower"], default=None),
            "anchored_vwap": _avwap(clean, internal, latest)}
