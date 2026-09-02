from __future__ import annotations

from typing import Any

import pandas as pd

import naked_k_patterns
import naked_k_portfolio


BULLISH_PATTERN_KEYS = ("看涨吸收", "看涨Pin", "锤子线", "早晨星", "蜻蜓十字", "阴孕阳")
BEARISH_PATTERN_KEYS = ("看跌吸收", "看跌Pin", "射击之星", "黄昏星", "墓碑十字", "阳孕阴")
WATCH_PATTERN_KEYS = ("十字星", "双孕线", "孕线")
BULLISH_ACTIONS = {"买入", "小仓试错"}
BEARISH_ACTIONS = {"减仓", "回避"}
MIN_ACTIONABLE_REWARD_TO_RISK = 1.0


def build_breakout_trigger(bar: pd.Series, side: str, buffer_ratio: float = 0.002) -> float:
    if side == "bullish":
        return round(float(bar["High"]) * (1 + buffer_ratio), 2)
    return round(float(bar["Low"]) * (1 - buffer_ratio), 2)


def build_invalidation_level(bar: pd.Series, side: str, buffer_ratio: float = 0.002) -> float:
    if side == "bullish":
        return round(float(bar["Low"]) * (1 - buffer_ratio), 2)
    return round(float(bar["High"]) * (1 + buffer_ratio), 2)


def calculate_atr(frame: pd.DataFrame, window: int = 14) -> float | None:
    if frame.empty or not {"High", "Low", "Close"}.issubset(frame.columns):
        return None
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.dropna()
    if true_range.empty:
        return None
    return float(true_range.tail(window).mean())


def build_volatility_buffer_ratio(
    frame: pd.DataFrame,
    base_ratio: float = 0.002,
    atr_fraction: float = 0.10,
    max_ratio: float = 0.015,
) -> float:
    atr = calculate_atr(frame)
    if atr is None:
        return base_ratio
    close = float(frame["Close"].iloc[-1])
    if close <= 0:
        return base_ratio
    atr_buffer = atr / close * atr_fraction
    return round(max(base_ratio, min(max_ratio, atr_buffer)), 4)


def build_signal_state(action: str) -> str:
    if action in BULLISH_ACTIONS:
        return "planned_long"
    if action in BEARISH_ACTIONS:
        return "planned_defensive"
    return "watching"


def build_trade_metrics(
    action: str,
    entry_trigger: float,
    stop_loss: float,
    resistance: float,
    support: float,
) -> tuple[float | None, float, float | None]:
    risk_per_share = round(abs(entry_trigger - stop_loss), 2)
    target_price: float | None = None
    if action in BULLISH_ACTIONS and resistance > entry_trigger:
        target_price = resistance
    # `减仓` / `回避` are defensive alerts, not short entries.

    reward_to_risk: float | None = None
    if target_price is not None and risk_per_share > 0:
        reward_to_risk = round(abs(target_price - entry_trigger) / risk_per_share, 2)
    return target_price, risk_per_share, reward_to_risk


def downgrade_low_reward_setup(
    action: str,
    target_price: float | None,
    reward_to_risk: float | None,
    minimum_reward_to_risk: float = MIN_ACTIONABLE_REWARD_TO_RISK,
) -> tuple[str, float | None, float | None, str | None]:
    if action not in BULLISH_ACTIONS:
        return action, target_price, reward_to_risk, None
    if target_price is None or reward_to_risk is None:
        return action, target_price, reward_to_risk, None
    if reward_to_risk >= minimum_reward_to_risk:
        return action, target_price, reward_to_risk, None
    return "观望", None, None, f"首个压力位空间不足，盈亏比不足 {minimum_reward_to_risk:.1f}R"


def build_position_guidance(
    action: str,
    entry_trigger: float,
    stop_loss: float,
    account_risk_pct: float = 1.0,
) -> str:
    if action == "买入":
        max_gross_pct = 30.0
    elif action == "小仓试错":
        max_gross_pct = 15.0
    elif action == "减仓":
        return "降至10%以内（仅处理已有多头，不新建仓）"
    elif action == "回避":
        return "不建仓"
    else:
        return "0%（无新仓计划）"

    if entry_trigger <= 0:
        return f"最高约{max_gross_pct:.1f}%仓位"
    risk_pct = abs(entry_trigger - stop_loss) / entry_trigger * 100
    if risk_pct <= 0:
        return f"最高约{max_gross_pct:.1f}%仓位"
    gross_pct = min(max_gross_pct, account_risk_pct / risk_pct * 100)
    return f"最高约{gross_pct:.1f}%仓位（按{account_risk_pct:g}%账户风险，单股风险{risk_pct:.1f}%）"


def _to_float(value: Any) -> float:
    return round(float(value), 2)


def _format_ts(value: Any, tz: str | None = None) -> str:
    """Render a bar timestamp, optionally converted to a market's local zone.

    Intraday frames are held in UTC so that Tencent's minute bars (naive Beijing
    at the source) and Yahoo's (naive UTC) can be compared on one axis. That makes
    UTC wrong for display: a bar closing 15:00 Beijing reads 07:00. The conversion
    belongs here, at the boundary, rather than in the frames.
    """
    timestamp = pd.Timestamp(value)
    if tz is None:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")


classify_market = naked_k_portfolio.classify_market


def _display_timezone(frame: pd.DataFrame, market: str | None) -> str | None:
    """Zone to render this frame's stamps in, or None to leave them raw.

    Read from the frame's own `attrs['ticker']` when the caller does not name a
    market. `download` records the ticker and it survives the reshape in
    `load_ohlcv`, so the frame carries everything needed — the same "label the
    frame, read it downstream" pattern used for `adjustment`. Deriving it here
    rather than threading a parameter means a caller cannot forget it: the
    post-news path in naked_k_synthesis did exactly that and silently reverted
    every `--news` run's intraday clock to UTC.
    """
    if market is None:
        ticker = frame.attrs.get("ticker")
        if not ticker:
            return None
        market = classify_market(str(ticker))
    if market == "crypto":
        # No local session to render against, so leave the stamp in UTC rather
        # than borrowing some exchange's clock.
        return None
    return naked_k_portfolio.market_timezone_name(market)


def build_intraday_status(
    frame: pd.DataFrame | None,
    action: str,
    entry_trigger: float,
    stop_loss: float,
    proximity_pct: float = 1.0,
    market: str | None = None,
) -> dict[str, Any]:
    if frame is None or getattr(frame, "empty", True):
        return {
            "status": "无盘中数据",
            "note": "未获取到1h盘中K线",
        }

    tz = _display_timezone(frame, market)
    latest = frame.iloc[-1]
    latest_volume = _to_float(latest.get("Volume", 0))
    payload = {
        "status": "盘中观察",
        "note": "未接近触发位或失效位",
        "source": str(frame.attrs.get("source", "unknown")),
        "interval": str(frame.attrs.get("interval", "1h")),
        "latest_time": _format_ts(frame.index[-1], tz),
        "timezone": tz or "UTC",
        "latest_close": _to_float(latest["Close"]),
        "latest_high": _to_float(latest["High"]),
        "latest_low": _to_float(latest["Low"]),
        "latest_volume": latest_volume,
    }

    if latest_volume <= 0:
        payload.update(
            {
                "status": "盘中数据未确认",
                "note": "最新1h K线成交量为0，等待有效K线确认",
            }
        )
        return payload

    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    proximity = proximity_pct / 100
    is_bearish_plan = action in BEARISH_ACTIONS

    if is_bearish_plan:
        if close <= entry_trigger:
            payload.update({"status": "盘中确认", "note": "最近有效1h收盘跌破触发位"})
        elif low <= entry_trigger:
            payload.update({"status": "盘中跌破未确认", "note": "盘中跌破触发位，但1h收盘未确认"})
        elif high >= stop_loss * (1 - proximity):
            payload.update({"status": "接近失效位", "note": "盘中价格接近空头失效位"})
        elif close <= entry_trigger * (1 + proximity):
            payload.update({"status": "接近触发", "note": "最新1h价格距离触发位1%以内"})
    else:
        if close >= entry_trigger:
            payload.update({"status": "盘中确认", "note": "最近有效1h收盘站上触发位"})
        elif high >= entry_trigger:
            payload.update({"status": "盘中突破未确认", "note": "盘中突破触发位，但1h收盘未确认"})
        elif low <= stop_loss * (1 + proximity):
            payload.update({"status": "接近失效位", "note": "盘中价格接近多头失效位"})
        elif close >= entry_trigger * (1 - proximity):
            payload.update({"status": "接近触发", "note": "最新1h价格距离触发位1%以内"})

    return payload


def _latest_bar_metrics(bar: pd.Series) -> dict[str, float]:
    open_price = float(bar["Open"])
    high = float(bar["High"])
    low = float(bar["Low"])
    close = float(bar["Close"])
    full_range = max(high - low, 0.0)
    body = abs(close - open_price)
    upper_shadow = max(high - max(open_price, close), 0.0)
    lower_shadow = max(min(open_price, close) - low, 0.0)
    if full_range <= 0:
        close_position_pct = 50.0
        body_pct = 0.0
        upper_shadow_pct = 0.0
        lower_shadow_pct = 0.0
    else:
        close_position_pct = (close - low) / full_range * 100
        body_pct = body / full_range * 100
        upper_shadow_pct = upper_shadow / full_range * 100
        lower_shadow_pct = lower_shadow / full_range * 100

    return {
        "open": round(open_price, 2),
        "high": round(high, 2),
        "low": round(low, 2),
        "close": round(close, 2),
        "range": round(full_range, 2),
        "body_pct": round(body_pct, 1),
        "upper_shadow_pct": round(upper_shadow_pct, 1),
        "lower_shadow_pct": round(lower_shadow_pct, 1),
        "close_position_pct": round(close_position_pct, 1),
    }


def classify_latest_candle(bar: pd.Series) -> list[str]:
    metrics = _latest_bar_metrics(bar)
    open_price = float(metrics["open"])
    close = float(metrics["close"])
    candle_range = float(metrics["range"])
    body_pct = float(metrics["body_pct"])
    upper_shadow_pct = float(metrics["upper_shadow_pct"])
    lower_shadow_pct = float(metrics["lower_shadow_pct"])
    close_position_pct = float(metrics["close_position_pct"])
    labels: list[str] = []

    if candle_range <= 0:
        return ["无波动K线"]

    if body_pct <= 12:
        labels.append("十字犹豫")
    elif close > open_price and body_pct >= 55 and close_position_pct >= 70:
        labels.append("强阳收近高点")
    elif close < open_price and body_pct >= 55 and close_position_pct <= 30:
        labels.append("强阴收近低点")

    if upper_shadow_pct >= 45 and upper_shadow_pct > lower_shadow_pct * 1.5 and close_position_pct < 60:
        labels.append("上影线压力")
    if lower_shadow_pct >= 45 and lower_shadow_pct > upper_shadow_pct * 1.5 and close_position_pct > 40:
        labels.append("下影线承接")

    if not labels:
        labels.append("普通阳线" if close >= open_price else "普通阴线")
    return labels


def analyze_trend_structure(clean: pd.DataFrame, window: int = 5) -> dict[str, Any]:
    recent = clean.tail(window)
    if len(recent) < 3:
        return {
            "direction": "neutral",
            "state": "样本不足",
            "strength": "weak",
            "score": 0,
        }

    highs = recent["High"].astype(float).tolist()
    lows = recent["Low"].astype(float).tolist()
    closes = recent["Close"].astype(float).tolist()
    comparisons = len(recent) - 1

    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    higher_closes = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
    lower_closes = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])

    up_points = higher_highs + higher_lows + higher_closes
    down_points = lower_highs + lower_lows + lower_closes
    max_points = comparisons * 3
    threshold = max(4, round(max_points * 0.65))

    if up_points >= threshold and up_points > down_points:
        direction = "up"
        state = "上升结构"
        raw_strength = up_points / max_points
        score = 1
    elif down_points >= threshold and down_points > up_points:
        direction = "down"
        state = "下降结构"
        raw_strength = down_points / max_points
        score = -1
    else:
        return {
            "direction": "sideways",
            "state": "横盘结构",
            "strength": "weak",
            "score": 0,
            "up_points": up_points,
            "down_points": down_points,
        }

    if raw_strength >= 0.85:
        strength = "strong"
    elif raw_strength >= 0.65:
        strength = "developing"
    else:
        strength = "weak"

    return {
        "direction": direction,
        "state": state,
        "strength": strength,
        "score": score,
        "up_points": up_points,
        "down_points": down_points,
    }


def analyze_pullback_context(clean: pd.DataFrame, lookback: int) -> dict[str, Any]:
    prior = clean.iloc[:-1].tail(lookback)
    if len(prior) < 3:
        return {"direction": "none", "zone": "样本不足"}

    latest_close = float(clean["Close"].iloc[-1])
    prior_high_idx = prior["High"].astype(float).idxmax()
    prior_low_idx = prior["Low"].astype(float).idxmin()
    prior_high = float(prior.loc[prior_high_idx, "High"])
    prior_low = float(prior.loc[prior_low_idx, "Low"])
    impulse = prior_high - prior_low
    if impulse <= 0:
        return {"direction": "none", "zone": "无有效波段"}

    if prior.index.get_loc(prior_low_idx) < prior.index.get_loc(prior_high_idx):
        direction = "bullish"
        depth_pct = (prior_high - latest_close) / impulse * 100
    elif prior.index.get_loc(prior_high_idx) < prior.index.get_loc(prior_low_idx):
        direction = "bearish"
        depth_pct = (latest_close - prior_low) / impulse * 100
    else:
        return {"direction": "none", "zone": "无有效波段"}

    if depth_pct < 0:
        zone = "突破延伸"
    elif depth_pct <= 23.6:
        zone = "浅回撤"
    elif depth_pct <= 50.0:
        zone = "健康回撤"
    elif depth_pct <= 61.8:
        zone = "深回撤观察"
    elif depth_pct <= 78.6:
        zone = "深回撤"
    else:
        zone = "趋势破坏"

    return {
        "direction": direction,
        "zone": zone,
        "depth_pct": round(depth_pct, 1),
        "anchor_low": round(prior_low, 2),
        "anchor_high": round(prior_high, 2),
    }


def classify_volatility_state(
    latest: pd.Series,
    prior: pd.DataFrame,
    prior_high: float,
    prior_low: float,
) -> dict[str, Any]:
    latest_range = float(latest["High"]) - float(latest["Low"])
    prior_ranges = (prior["High"].astype(float) - prior["Low"].astype(float)).tail(5)
    avg_range = float(prior_ranges.mean()) if not prior_ranges.empty else 0.0
    if latest_range <= 0 or avg_range <= 0:
        return {"state": "波动未知", "range_ratio": None}

    range_ratio = latest_range / avg_range
    latest_close = float(latest["Close"])
    if range_ratio >= 1.5 and latest_close > prior_high:
        state = "突破扩张"
    elif range_ratio >= 1.5 and latest_close < prior_low:
        state = "跌破扩张"
    elif range_ratio >= 1.5:
        state = "宽幅震荡"
    elif range_ratio <= 0.7:
        state = "波幅压缩"
    else:
        state = "常态波动"

    return {
        "state": state,
        "range_ratio": round(range_ratio, 2),
    }


def analyze_price_action_context(frame: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if frame.empty or len(frame) < 2:
        return {
            "bias": "neutral",
            "candle": [],
            "signals": ["样本不足"],
            "warnings": [],
        }

    clean = frame[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(clean) < 2:
        return {
            "bias": "neutral",
            "candle": [],
            "signals": ["样本不足"],
            "warnings": [],
        }

    latest = clean.iloc[-1]
    prior = clean.iloc[:-1].tail(lookback)
    actual_lookback = len(prior)
    prior_high = float(prior["High"].max())
    prior_low = float(prior["Low"].min())
    latest_high = float(latest["High"])
    latest_low = float(latest["Low"])
    latest_close = float(latest["Close"])
    metrics = _latest_bar_metrics(latest)
    candle = classify_latest_candle(latest)
    signals: list[str] = []
    warnings: list[str] = []
    score = 0
    trend = analyze_trend_structure(clean)
    pullback = analyze_pullback_context(clean, actual_lookback)
    volatility = classify_volatility_state(latest, prior, prior_high, prior_low)

    if latest_high > prior_high and latest_close < prior_high:
        signals.append(f"上破{actual_lookback}日高点失败")
        warnings.append("前高假突破风险")
        score -= 2
    elif latest_close > prior_high:
        signals.append(f"收盘突破{actual_lookback}日高点")
        score += 2

    if latest_low < prior_low and latest_close > prior_low:
        signals.append(f"下破{actual_lookback}日低点收回")
        score += 2
    elif latest_close < prior_low:
        signals.append(f"收盘跌破{actual_lookback}日低点")
        warnings.append("前低失守")
        score -= 2

    if trend["direction"] == "up":
        signals.append("趋势结构向上")
        score += int(trend["score"])
    elif trend["direction"] == "down":
        signals.append("趋势结构向下")
        score += int(trend["score"])

    volatility_state = str(volatility["state"])
    if volatility_state == "突破扩张":
        score += 1
    elif volatility_state == "跌破扩张":
        score -= 1

    recent = clean.tail(3)
    if len(recent) == 3:
        highs = recent["High"].astype(float).tolist()
        lows = recent["Low"].astype(float).tolist()
        if highs[0] < highs[1] < highs[2] and lows[0] < lows[1] < lows[2]:
            signals.append("高低点抬高")
            score += 1
        elif highs[0] > highs[1] > highs[2] and lows[0] > lows[1] > lows[2]:
            signals.append("高低点降低")
            score -= 1

    ranges = (clean["High"].astype(float) - clean["Low"].astype(float)).tail(4).tolist()
    if len(ranges) == 4 and ranges[1] > ranges[2] > ranges[3] and ranges[3] < sum(ranges[:3]) / 3:
        signals.append("波幅连续收敛")

    if "强阳收近高点" in candle:
        score += 1
    if "强阴收近低点" in candle:
        score -= 1
    if "上影线压力" in candle:
        score -= 1
    if "下影线承接" in candle:
        score += 1

    latest_volume = float(latest.get("Volume", 0) or 0)
    avg_volume = float(prior["Volume"].astype(float).mean()) if "Volume" in prior else 0.0
    volume_state = "成交量中性"
    if avg_volume > 0 and latest_volume >= avg_volume * 1.5:
        volume_state = "放量"
        if score > 0:
            signals.append("放量配合多头K线")
        elif score < 0:
            signals.append("放量配合空头K线")
        else:
            signals.append("放量换手")
    elif avg_volume > 0 and latest_volume <= avg_volume * 0.6:
        volume_state = "缩量"

    volume_pressure = "量能中性"
    if volume_state == "放量":
        if latest_high > prior_high and latest_close < prior_high:
            volume_pressure = "派发压力"
            warnings.append("放量上破失败")
            score -= 1
        elif latest_low < prior_low and latest_close > prior_low:
            volume_pressure = "承接增强"
            signals.append("放量下破收回")
            score += 1
        elif volatility_state == "突破扩张" and metrics["close_position_pct"] >= 70:
            volume_pressure = "量价确认"
            signals.append("放量突破扩张")
            signals.append("量价确认")
            score += 1
        elif volatility_state == "跌破扩张" and metrics["close_position_pct"] <= 30:
            volume_pressure = "量价确认"
            signals.append("放量跌破扩张")
            signals.append("量价确认")
            score -= 1
        elif float(latest["Close"]) > float(clean["Close"].iloc[-2]) and metrics["close_position_pct"] >= 65:
            volume_pressure = "量价确认"
            signals.append("量价确认")
        elif float(latest["Close"]) < float(clean["Close"].iloc[-2]) and metrics["close_position_pct"] <= 35:
            volume_pressure = "量价确认"
            signals.append("量价确认")
    elif volume_state == "缩量":
        if latest_close > prior_high:
            volume_pressure = "缩量突破待确认"
            warnings.append("缩量突破，等待放量确认")
        elif latest_close < prior_low:
            volume_pressure = "缩量跌破待确认"
            warnings.append("缩量跌破，等待放量确认")

    if score >= 2:
        bias = "bullish"
    elif score <= -2:
        bias = "bearish"
    elif any(signal in signals for signal in ["波幅连续收敛"]) or "十字犹豫" in candle:
        bias = "watch"
    else:
        bias = "neutral"

    if not signals:
        signals.append("区间内震荡")

    return {
        "bias": bias,
        "score": score,
        "candle": candle,
        "signals": signals,
        "warnings": warnings,
        "lookback": actual_lookback,
        "range_high": round(prior_high, 2),
        "range_low": round(prior_low, 2),
        "close_position_pct": metrics["close_position_pct"],
        "body_pct": metrics["body_pct"],
        "upper_shadow_pct": metrics["upper_shadow_pct"],
        "lower_shadow_pct": metrics["lower_shadow_pct"],
        "volume_state": volume_state,
        "volume_pressure": volume_pressure,
        "trend": trend,
        "volatility_state": volatility_state,
        "volatility": volatility,
        "pullback": pullback,
    }


def format_price_action_summary(price_action: dict[str, Any]) -> str:
    if not price_action:
        return "暂无"
    parts: list[str] = []
    candle = price_action.get("candle") or []
    signals = price_action.get("signals") or []
    warnings = price_action.get("warnings") or []
    trend = price_action.get("trend") or {}
    pullback = price_action.get("pullback") or {}
    if candle:
        parts.append(f"K线：{'、'.join(str(item) for item in candle)}")
    if signals:
        parts.append(f"结构：{'、'.join(str(item) for item in signals)}")
    if trend.get("state") and trend.get("state") != "样本不足":
        strength = {"strong": "强", "developing": "形成中", "weak": "弱"}.get(
            str(trend.get("strength")),
            str(trend.get("strength")),
        )
        parts.append(f"趋势：{trend['state']}（{strength}）")
    if price_action.get("volatility_state"):
        parts.append(f"波动：{price_action['volatility_state']}")
    if pullback.get("direction") not in {None, "none"} and pullback.get("zone"):
        if pullback.get("depth_pct") is not None:
            parts.append(f"回撤：{pullback['zone']}（{pullback['depth_pct']}%）")
        else:
            parts.append(f"回撤：{pullback['zone']}")
    if warnings:
        parts.append(f"风险：{'、'.join(str(item) for item in warnings)}")
    if price_action.get("close_position_pct") is not None:
        parts.append(f"收盘位置：{price_action['close_position_pct']}%")
    if price_action.get("volume_pressure"):
        parts.append(f"量价：{price_action['volume_pressure']}")
    if price_action.get("volume_state"):
        parts.append(f"量能：{price_action['volume_state']}")
    return "；".join(parts) if parts else "暂无"


def format_market_structure_summary(market_structure: dict[str, Any]) -> str:
    if not market_structure:
        return "暂无"

    direction_label = {
        "up": "上升结构",
        "down": "下降结构",
        "transition": "结构转换",
        "expanding_range": "扩张震荡",
        "contracting_range": "收敛震荡",
        "sideways": "横盘结构",
        "neutral": "结构未确认",
    }.get(str(market_structure.get("direction")), str(market_structure.get("direction", "结构未确认")))
    parts = [direction_label]

    sequence = market_structure.get("sequence")
    if sequence and sequence != "样本不足":
        parts.append(f"序列：{sequence}")

    event = market_structure.get("latest_event")
    if event:
        event_direction = "向上" if event.get("direction") == "bullish" else "向下"
        parts.append(f"{event.get('kind')} {event_direction}突破 {event.get('broken_level')}")

    last_high = market_structure.get("last_swing_high")
    last_low = market_structure.get("last_swing_low")
    if last_high:
        parts.append(f"最近结构高点：{last_high.get('price')}")
    if last_low:
        parts.append(f"最近结构低点：{last_low.get('price')}")

    return "；".join(parts)


def format_market_regime_summary(market_regime: dict[str, Any]) -> str:
    if not market_regime:
        return "暂无"
    label = str(market_regime.get("label") or "状态未知")
    direction = str(market_regime.get("direction") or "neutral")
    direction_label = {
        "bullish": "偏多",
        "bearish": "偏空",
        "neutral": "中性",
    }.get(direction, direction)
    range_ratio = market_regime.get("range_ratio")
    if range_ratio is None:
        return f"{label}（方向：{direction_label}）"
    return f"{label}（方向：{direction_label}，波幅比：{range_ratio}）"


def format_risk_plan_summary(risk_plan: dict[str, Any]) -> str:
    if not risk_plan:
        return "暂无"

    status_label = {
        "active": "可执行",
        "reduced": "降风险",
        "blocked": "暂停新仓",
        "flat": "无新仓",
    }.get(str(risk_plan.get("status")), str(risk_plan.get("status", "未知")))
    direction_label = {
        "long": "多头",
        "short": "空头/风控",
        "bearish_defensive": "防御/回避",
        "none": "无方向",
    }.get(str(risk_plan.get("direction")), str(risk_plan.get("direction", "无方向")))
    parts = [
        f"{status_label}",
        f"方向：{direction_label}",
        f"账户风险：{risk_plan.get('effective_account_risk_pct', 0)}%",
        f"建议仓位：{risk_plan.get('suggested_gross_pct', 0)}%",
        f"单笔风险：{risk_plan.get('risk_pct', 0)}%",
    ]
    target_r = risk_plan.get("target_r_multiple")
    if target_r is not None:
        parts.append(f"目标：{target_r}R")
    targets = risk_plan.get("targets_by_r") or {}
    if targets:
        parts.append("R目标：" + " / ".join(f"{label}={price}" for label, price in targets.items()))
    guardrails = risk_plan.get("guardrails") or []
    if guardrails:
        parts.append("保护：" + "、".join(str(item) for item in guardrails))
    return "；".join(parts)


def format_trade_setup_summary(trade_setup: dict[str, Any]) -> str:
    if not trade_setup:
        return "暂无"

    direction_label = {
        "long": "多头",
        "short": "空头/风控",
        "watch": "观察",
        "none": "无方向",
    }.get(str(trade_setup.get("direction")), str(trade_setup.get("direction", "无方向")))
    parts = [
        str(trade_setup.get("name") or trade_setup.get("key") or "未命名剧本"),
        f"方向：{direction_label}",
        f"质量：{trade_setup.get('quality', 'unknown')}",
        f"置信：{trade_setup.get('confidence_score', 0)}",
    ]
    thesis = trade_setup.get("thesis")
    if thesis:
        parts.append(f"逻辑：{thesis}")
    confirmations = trade_setup.get("required_confirmation") or []
    if confirmations:
        parts.append("确认：" + "、".join(str(item) for item in confirmations))
    invalidation = trade_setup.get("invalidation_logic")
    if invalidation:
        parts.append(f"失效：{invalidation}")
    return "；".join(parts)


def format_price_zones_summary(price_zones: dict[str, Any]) -> str:
    if not price_zones:
        return "暂无"

    parts: list[str] = []
    nearest_support = price_zones.get("nearest_support")
    nearest_resistance = price_zones.get("nearest_resistance")
    if nearest_support:
        parts.append(
            f"支撑/需求：{nearest_support.get('label')} {nearest_support.get('lower')}-{nearest_support.get('upper')}"
            f"（{nearest_support.get('strength')}，触碰{nearest_support.get('touches')}次）"
        )
    if nearest_resistance:
        parts.append(
            f"压力/供给：{nearest_resistance.get('label')} {nearest_resistance.get('lower')}-{nearest_resistance.get('upper')}"
            f"（{nearest_resistance.get('strength')}，触碰{nearest_resistance.get('touches')}次）"
        )

    liquidity_pools = price_zones.get("liquidity_pools") or []
    if liquidity_pools:
        pool = liquidity_pools[0]
        parts.append(f"流动性：{pool.get('label')} {pool.get('lower')}-{pool.get('upper')}")

    volume_zones = price_zones.get("volume_zones") or []
    if volume_zones:
        zone = volume_zones[0]
        parts.append(f"成交密集：{zone.get('lower')}-{zone.get('upper')}（{zone.get('strength')}）")

    volume_profile = price_zones.get("volume_profile") or {}
    poc = volume_profile.get("poc") or {}
    value_area = volume_profile.get("value_area") or {}
    if poc:
        parts.append(f"POC：{poc.get('midpoint')}（{poc.get('lower')}-{poc.get('upper')}）")
    if value_area:
        parts.append(
            f"价值区域：{value_area.get('lower')}-{value_area.get('upper')}"
            f"（量占比{value_area.get('volume_share')}）"
        )

    anchored_vwap = price_zones.get("anchored_vwap") or {}
    if anchored_vwap:
        side = "下方" if anchored_vwap.get("side") == "below" else "上方"
        parts.append(
            f"Anchored VWAP：{anchored_vwap.get('value')}（{side}，锚点{anchored_vwap.get('anchor_type')}）"
        )

    return "；".join(parts) if parts else "暂无"


def classify_patterns(patterns: list[str]) -> str:
    joined = " ".join(patterns)
    if any(key in joined for key in BULLISH_PATTERN_KEYS):
        return "bullish"
    if any(key in joined for key in BEARISH_PATTERN_KEYS):
        return "bearish"
    if any(key in joined for key in WATCH_PATTERN_KEYS):
        return "watch"
    return "neutral"


def detect_price_action_patterns(frame: pd.DataFrame) -> list[str]:
    patterns = list(naked_k_patterns.detect_kline_patterns(frame))
    inside = naked_k_patterns.detect_inside_bar(frame)
    if inside:
        patterns.append(inside)
    return patterns


def resolve_weekly_context(frame: pd.DataFrame, patterns: list[str]) -> str:
    latest = frame.iloc[-1]
    recent = frame.tail(9)
    prior = recent.iloc[:-1]
    pattern_bias = classify_patterns(patterns)
    if pattern_bias == "bullish":
        return "周线偏多，允许日线多头触发"
    if pattern_bias == "bearish":
        return "周线偏空，优先过滤日线追多"

    prior_high = float(prior["High"].max()) if not prior.empty else float(latest["High"])
    prior_low = float(prior["Low"].min()) if not prior.empty else float(latest["Low"])
    range_mid = (prior_high + prior_low) / 2
    if float(latest["Close"]) >= range_mid:
        return "周线中性偏多，只接受确认后做多"
    return "周线中性偏空，等待更强确认"


def find_price_levels(frame: pd.DataFrame, close: float) -> tuple[float, float]:
    recent = frame.tail(30)
    swing_lows: list[float] = []
    swing_highs: list[float] = []
    for i in range(1, len(recent) - 1):
        low = float(recent["Low"].iloc[i])
        high = float(recent["High"].iloc[i])
        if low <= float(recent["Low"].iloc[i - 1]) and low <= float(recent["Low"].iloc[i + 1]):
            swing_lows.append(low)
        if high >= float(recent["High"].iloc[i - 1]) and high >= float(recent["High"].iloc[i + 1]):
            swing_highs.append(high)

    support_candidates = [value for value in swing_lows if value < close]
    resistance_candidates = [value for value in swing_highs if value > close]

    support = max(support_candidates) if support_candidates else float(recent["Low"].tail(10).min())
    resistance = min(resistance_candidates) if resistance_candidates else float(recent["High"].tail(10).max())
    return round(support, 2), round(resistance, 2)


def review_previous_call(previous: dict[str, Any] | None, current_bar: pd.Series, current_close: float) -> dict[str, Any]:
    if not previous:
        return {"status": "无上次记录", "error_type": None, "note": "首日运行，暂无可复盘样本"}

    action = previous.get("action")
    trigger = previous.get("entry_trigger")
    stop_loss = previous.get("stop_loss")
    high = float(current_bar["High"])
    low = float(current_bar["Low"])

    if action in BULLISH_ACTIONS:
        if trigger is not None and high >= float(trigger):
            if stop_loss is not None and low <= float(stop_loss):
                return {"status": "未命中", "error_type": "假突破", "note": "触发后回落到失效位"}
            if current_close < float(trigger):
                return {"status": "未命中", "error_type": "假突破", "note": "盘中上破但收盘未站稳触发位"}
            return {"status": "命中", "error_type": None, "note": "触发位被突破且收盘保持在其上"}
        return {"status": "未触发", "error_type": "缺少确认K", "note": "没有突破信号K高点"}

    if action in BEARISH_ACTIONS:
        if trigger is not None and low <= float(trigger):
            if stop_loss is not None and high >= float(stop_loss):
                return {"status": "未命中", "error_type": "假跌破", "note": "跌破后快速收回失效位"}
            if current_close > float(trigger):
                return {"status": "未命中", "error_type": "假跌破", "note": "盘中跌破但收盘未守住"}
            return {"status": "命中", "error_type": None, "note": "跌破触发位且收盘仍弱"}
        return {"status": "未触发", "error_type": "缺少确认K", "note": "没有跌破信号K低点"}

    return {"status": "观察中", "error_type": None, "note": "上一交易日偏观察，不计入成败"}
