from __future__ import annotations

from typing import Any


def _value(report: Any, field: str, default: Any = None) -> Any:
    if isinstance(report, dict):
        return report.get(field, default)
    return getattr(report, field, default)


def _as_dict(report: Any, field: str) -> dict[str, Any]:
    value = _value(report, field, {})
    return value if isinstance(value, dict) else {}


def _direction_text(direction: str) -> str:
    return {
        "long": "多头",
        "short": "空头/风控",
        "watch": "观察",
        "bullish": "偏多",
        "bearish": "偏空",
        "neutral": "中性",
    }.get(direction, direction)


def _zone_text(zone: dict[str, Any] | None) -> str:
    if not zone:
        return "暂无"
    label = zone.get("label", "区域")
    lower = zone.get("lower")
    upper = zone.get("upper")
    if lower is None or upper is None:
        return str(label)
    return f"{label} {lower}-{upper}"


def _format_smart_money_brief(smart_money: dict[str, Any]) -> str:
    """格式化 OHLCV 量价代理简报。"""
    if not smart_money or not smart_money.get("enabled"):
        return "暂无量价代理信号"

    assessment = smart_money.get("overall_assessment", "")

    # 优先使用 fresh_signals，如果没有则过滤 stale
    fresh_signals = smart_money.get("fresh_signals")
    if fresh_signals is None:
        # 向后兼容：手动过滤
        all_signals = smart_money.get("signals", [])
        fresh_signals = [s for s in all_signals if not s.get("stale", False)]

    stale_signals = smart_money.get("stale_signals", [])

    if not fresh_signals and not stale_signals:
        return assessment or "暂无量价代理信号"

    if not fresh_signals:
        # 只有过期信号
        return f"无新鲜量价代理信号（{len(stale_signals)}个信号已过期）"

    # 提取关键的新鲜信号
    signal_texts = []
    for signal in fresh_signals[:3]:  # 最多显示3个
        label = signal.get("label", "")
        strength = signal.get("strength", "")
        days_old = signal.get("days_old", 0)

        if days_old > 0:
            signal_texts.append(f"{label}({strength}, {days_old}日前)")
        else:
            signal_texts.append(f"{label}({strength})")

    result = f"{assessment}。检测到：{' + '.join(signal_texts)}"

    # 如果有过期信号，追加说明
    if stale_signals:
        result += f"；另有{len(stale_signals)}个过期信号"

    return result


def build_trader_brief(report: Any) -> dict[str, Any]:
    price_action = _as_dict(report, "price_action")
    structure = _as_dict(report, "market_structure")
    regime = _as_dict(report, "market_regime")
    setup = _as_dict(report, "trade_setup")
    zones = _as_dict(report, "price_zones")
    risk_plan = _as_dict(report, "risk_plan")
    timeframe = _as_dict(report, "timeframe_context")
    smart_money = _as_dict(report, "smart_money_signals")

    action = str(_value(report, "action", "观望"))
    direction = _direction_text(str(setup.get("direction", "watch")))
    signals = "、".join(str(item) for item in price_action.get("signals", [])[:4]) or "暂无明确结构信号"
    warnings = [str(item) for item in price_action.get("warnings", [])]
    event = structure.get("latest_event") or {}
    event_text = (
        f"{event.get('kind')} {event.get('direction')} 打穿 {event.get('broken_level')}"
        if event
        else "暂无最新 BOS/CHoCH"
    )
    support = _zone_text(zones.get("nearest_support"))
    resistance = _zone_text(zones.get("nearest_resistance"))
    thesis = str(setup.get("thesis") or "等待价格在关键区域给出确认")
    decision_filter = str(timeframe.get("decision_filter") or "多周期证据不足，按低风险观察处理")

    target = _value(report, "target_price")
    reward_to_risk = _value(report, "reward_to_risk")
    reward_text = f"{reward_to_risk}R" if reward_to_risk is not None else "暂无"
    entry_trigger = round(float(_value(report, "entry_trigger", 0.0)), 2)
    stop_loss = round(float(_value(report, "stop_loss", 0.0)), 2)
    invalidation_verb = "跌破" if stop_loss < entry_trigger else "突破" if stop_loss > entry_trigger else "触及"
    continuation_path = (
        f"路径A：价格触发 {entry_trigger} 后延续，先看 {target}"
        if target is not None
        else f"路径A：价格触发 {entry_trigger} 后，等待收盘确认再评估"
    )

    # OHLCV 量价代理分析
    smart_money_text = _format_smart_money_brief(smart_money)

    risk_points = warnings[:]
    if timeframe.get("alignment") == "conflict":
        risk_points.append("日线机会与高周期方向冲突")
    if not risk_points:
        risk_points.append(f"若触发后不能延续并{invalidation_verb}失效位，说明结构判断失败")

    return {
        "当前市场状态": (
            f"{regime.get('label', '状态未知')}；结构事件：{event_text}；"
            f"多周期过滤：{decision_filter}"
        ),
        "多空力量分析": (
            f"{signals}。{thesis}；当前剧本方向为{direction}。"
            f"量价状态：{price_action.get('volume_pressure', '量能中性')}"
        ),
        "量价代理证据": smart_money_text,
        "关键价格区域": f"下方：{support}；上方：{resistance}",
        "可能交易路径": [
            continuation_path,
            f"路径B：触发失败或{invalidation_verb}失效位 {stop_loss}，计划作废",
            "路径C：未触发则继续观察，避免在区间中部追价",
        ],
        "交易计划": (
            f"当前机会：{action}；盈亏比：{reward_text}；"
            f"建议仓位：{_value(report, 'position_size', '暂无')}；"
            f"失效位置：{stop_loss}；"
            f"风险等级：{risk_plan.get('risk_level', 'medium')}"
        ),
        "风险点": risk_points,
    }


def format_trader_brief(brief: dict[str, Any]) -> str:
    if not brief:
        return "暂无"
    paths = brief.get("可能交易路径") or []
    path_text = " / ".join(str(item) for item in paths)
    risks = brief.get("风险点") or []
    risk_text = "、".join(str(item) for item in risks)
    return (
        f"状态：{brief.get('当前市场状态', '暂无')}；"
        f"力量：{brief.get('多空力量分析', '暂无')}；"
        f"量价代理：{brief.get('量价代理证据', brief.get('主力行为研判', '暂无'))}；"
        f"区域：{brief.get('关键价格区域', '暂无')}；"
        f"路径：{path_text or '暂无'}；"
        f"计划：{brief.get('交易计划', '暂无')}；"
        f"风险：{risk_text or '暂无'}"
    )
