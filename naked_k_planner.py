from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

import naked_k_ai
import naked_k_config
import naked_k_context
import naked_k_interpreter
import naked_k_price_evidence
import naked_k_risk
import naked_k_setups
import naked_k_smart_money
import naked_k_smart_money_fusion
import naked_k_structure
import naked_k_trade
import naked_k_timeframes
import naked_k_zones


BULLISH_ACTIONS = {"买入", "小仓试错"}
BEARISH_ACTIONS = {"减仓", "回避"}


@dataclass
class InstrumentReport:
    name: str
    ticker: str
    action: str
    entry_trigger: float
    stop_loss: float
    target_price: float | None
    risk_per_share: float
    reward_to_risk: float | None
    signal_state: str
    resistance: float
    support: float
    position_size: str
    rationale: str
    daily_patterns: list[str]
    weekly_patterns: list[str]
    weekly_context: str
    data_sources: dict[str, str]
    latest_k_dates: dict[str, str]
    latest_closes: dict[str, float]
    review: dict[str, Any]
    improvement: str
    intraday_status: dict[str, Any] = field(default_factory=dict)
    price_action: dict[str, Any] = field(default_factory=dict)
    market_structure: dict[str, Any] = field(default_factory=dict)
    market_regime: dict[str, Any] = field(default_factory=dict)
    risk_plan: dict[str, Any] = field(default_factory=dict)
    trade_setup: dict[str, Any] = field(default_factory=dict)
    price_zones: dict[str, Any] = field(default_factory=dict)
    timeframe_context: dict[str, Any] = field(default_factory=dict)
    trader_brief: dict[str, Any] = field(default_factory=dict)
    candle_context: list[dict[str, Any]] = field(default_factory=list)
    ai_assistant: dict[str, Any] = field(default_factory=dict)
    smart_money_signals: dict[str, Any] = field(default_factory=dict)
    dual_evidence_fusion: dict[str, Any] | None = None
    price_evidences: list[dict[str, Any]] = field(default_factory=list)
    trade_flow_evidences: list[dict[str, Any]] = field(default_factory=list)
    technical_conclusion: dict[str, Any] = field(default_factory=dict)
    news_analysis: dict[str, Any] = field(default_factory=dict)
    combined_conclusion: dict[str, Any] = field(default_factory=dict)


def refresh_report_derivatives(report: InstrumentReport) -> None:
    llm_commentary = report.ai_assistant.get("llm_commentary")
    report.trader_brief = naked_k_interpreter.build_trader_brief(report)
    report.ai_assistant = naked_k_ai.build_ai_trading_assistant(report)
    if llm_commentary is not None:
        report.ai_assistant["llm_commentary"] = llm_commentary


def build_trade_plan(
    name: str,
    ticker: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    previous: dict[str, Any] | None,
    intraday: pd.DataFrame | None = None,
    monthly: pd.DataFrame | None = None,
    config: naked_k_config.TradingConfig | None = None,
) -> InstrumentReport:
    daily_bar = daily.iloc[-1]
    weekly_bar = weekly.iloc[-1]
    daily_patterns = naked_k_trade.detect_price_action_patterns(daily)
    weekly_patterns = naked_k_trade.detect_price_action_patterns(weekly)
    price_action = naked_k_trade.analyze_price_action_context(daily)
    market_structure = naked_k_structure.analyze_market_structure(daily, swing_window=1)
    market_regime = naked_k_structure.classify_market_regime(daily, market_structure)
    trade_setup = naked_k_setups.classify_trade_setup(
        price_action=price_action,
        market_structure=market_structure,
        market_regime=market_regime,
        daily_patterns=daily_patterns,
    )
    weekly_context = naked_k_trade.resolve_weekly_context(weekly, weekly_patterns)
    fallback_support, fallback_resistance = naked_k_trade.find_price_levels(daily, float(daily_bar["Close"]))
    price_zones = naked_k_zones.detect_price_zones(daily, close=float(daily_bar["Close"]), swing_window=1)
    candle_context = naked_k_context.build_candle_behavior_context(
        daily,
        price_action=price_action,
        market_structure=market_structure,
        price_zones=price_zones,
    )
    nearest_support = price_zones.get("nearest_support")
    nearest_resistance = price_zones.get("nearest_resistance")
    support = round(float(nearest_support["midpoint"]), 2) if nearest_support else fallback_support
    resistance = round(float(nearest_resistance["midpoint"]), 2) if nearest_resistance else fallback_resistance
    buffer_ratio = naked_k_trade.build_volatility_buffer_ratio(daily)

    daily_bias = naked_k_trade.classify_patterns(daily_patterns)
    weekly_bias = naked_k_trade.classify_patterns(weekly_patterns)
    structure_bias = str(price_action.get("bias", "neutral"))
    structure_event = market_structure.get("latest_event") or {}
    structure_event_bias = str(structure_event.get("direction", "neutral"))
    setup_direction = str(trade_setup.get("direction", "watch"))

    if daily_bias == "bullish":
        action = "买入" if weekly_bias == "bullish" else "小仓试错"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bullish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bullish", buffer_ratio=buffer_ratio)
    elif daily_bias == "bearish":
        action = "回避" if weekly_bias in {"bearish", "neutral"} else "减仓"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bearish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bearish", buffer_ratio=buffer_ratio)
    elif daily_bias == "watch":
        action = "观望"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bullish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bearish", buffer_ratio=buffer_ratio)
    elif structure_bias == "bullish":
        action = "买入" if weekly_bias == "bullish" else "小仓试错"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bullish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bullish", buffer_ratio=buffer_ratio)
    elif structure_bias == "bearish":
        action = "回避" if weekly_bias in {"bearish", "neutral"} else "减仓"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bearish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bearish", buffer_ratio=buffer_ratio)
    elif setup_direction == "long":
        action = "买入" if weekly_bias == "bullish" else "小仓试错"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bullish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bullish", buffer_ratio=buffer_ratio)
    elif setup_direction == "short":
        action = "回避" if weekly_bias in {"bearish", "neutral"} else "减仓"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bearish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bearish", buffer_ratio=buffer_ratio)
    elif structure_event_bias == "bullish":
        action = "买入" if weekly_bias == "bullish" else "小仓试错"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bullish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bullish", buffer_ratio=buffer_ratio)
    elif structure_event_bias == "bearish":
        action = "回避" if weekly_bias in {"bearish", "neutral"} else "减仓"
        entry_trigger = naked_k_trade.build_breakout_trigger(daily_bar, "bearish", buffer_ratio=buffer_ratio)
        stop_loss = naked_k_trade.build_invalidation_level(daily_bar, "bearish", buffer_ratio=buffer_ratio)
    else:
        action = "观望"
        entry_trigger = round(resistance * 1.002, 2)
        stop_loss = round(support * 0.998, 2)

    target_price, risk_per_share, reward_to_risk = naked_k_trade.build_trade_metrics(
        action,
        entry_trigger,
        stop_loss,
        resistance,
        support,
    )
    action, target_price, reward_to_risk, reward_filter_note = naked_k_trade.downgrade_low_reward_setup(
        action,
        target_price,
        reward_to_risk,
    )
    risk_plan = naked_k_risk.build_risk_plan(
        action=action,
        entry_trigger=entry_trigger,
        stop_loss=stop_loss,
        target_price=target_price,
        config=config.risk if config is not None else None,
    )
    position_size = (
        str(risk_plan["position_size"])
        if action in BULLISH_ACTIONS
        else naked_k_trade.build_position_guidance(action, entry_trigger, stop_loss)
    )
    signal_state = naked_k_trade.build_signal_state(action)
    intraday_status = naked_k_trade.build_intraday_status(
        intraday,
        action,
        entry_trigger,
        stop_loss,
        market=naked_k_trade.classify_market(ticker),
    )
    timeframe_context = naked_k_timeframes.build_timeframe_context(
        monthly=monthly,
        weekly=weekly,
        daily=daily,
        intraday_status=intraday_status,
        daily_price_action=price_action,
        daily_structure=market_structure,
        daily_regime=market_regime,
    )

    # OHLCV 量价代理分析
    monthly_zones = naked_k_zones.detect_price_zones(monthly, close=float(monthly["Close"].iloc[-1]), swing_window=2) if monthly is not None and not monthly.empty else None
    weekly_zones = naked_k_zones.detect_price_zones(weekly, close=float(weekly_bar["Close"]), swing_window=1)

    # 获取配置
    smart_money_config = getattr(config, 'smart_money', None) if config else None

    smart_money_signals = naked_k_smart_money.analyze_smart_money_signals(
        daily_df=daily,
        zones=price_zones.get("zones", []),
        liquidity_pools=price_zones.get("liquidity_pools", []),
        market_structure=market_structure,
        monthly_zones=monthly_zones.get("zones") if monthly_zones else None,
        weekly_zones=weekly_zones.get("zones"),
        config=smart_money_config,
    )

    # Dual-evidence 架构分析
    dual_evidence_fusion = None
    price_evidences_list = []
    trade_flow_evidences_list = []

    try:
        # 1. 生成价格证据层
        from naked_k_price_evidence import PriceActionEvidenceConfig

        price_config = PriceActionEvidenceConfig()
        decision_time = pd.Timestamp.now()

        price_action_layer = naked_k_price_evidence.build_price_action_layer(
            daily,
            zones=price_zones.get("zones", []),
            liquidity_pools=price_zones.get("liquidity_pools", []),
            market_structure=market_structure,
            patterns=daily_patterns,
            decision_time=decision_time,
            config=price_config,
        )

        if price_action_layer.availability == "available":
            price_evidences_list = [naked_k_price_evidence.evidence_to_dict(e) for e in price_action_layer.evidence]

            # participation state 是融合层新增的信息，必须算；direction 不能重算——
            # layer.direction 已经过了 30 天窗口过滤，而 evidence 故意保留全部历史，
            # 按全量重投票会把过期信号投回来（见 build_price_action_layer）。
            price_state, _ = naked_k_smart_money_fusion._compute_layer_state(
                price_action_layer.evidence,
                quality=price_action_layer.quality,
                limitations=price_action_layer.limitations,
            )
            price_layer = naked_k_smart_money_fusion.LayerResult(
                layer="price_action",
                state=price_state,
                direction=price_action_layer.direction,
                evidences=tuple(price_action_layer.evidence),
                quality=price_action_layer.quality,
                limitations=price_action_layer.limitations,
                decision_time=price_action_layer.as_of,
                target_session=daily.index[-1].strftime('%Y-%m-%d'),
                valid_from=price_action_layer.valid_from,
                valid_until=price_action_layer.expires_at,
            )
        else:
            price_layer = None

        # 2. 获取分钟线聚合资金流（替代失效的东财逐笔接口）
        trade_flow_layer = None
        intraday_snapshot = None
        if ticker.endswith('.HK'):
            try:
                import naked_k_intraday_flow
                intraday_snapshot = naked_k_intraday_flow.collect_intraday_flow(ticker)
                
                # 分钟线聚合只提供 snapshot 展示，不生成独立 evidence
                # 因为与日线 Volume 属同一依赖组（按设计文档 §8.5）
                # trade_flow_layer 保持 None，fusion 将为单层 price_action
                
                # 记录到 trade_flow_evidences_list 用于报告显示
                if intraday_snapshot.status == "OK":
                    trade_flow_evidences_list = [{
                        "kind": "intraday_volume_distribution",
                        "quality": intraday_snapshot.quality,
                        "limitations": list(intraday_snapshot.limitations),
                        "snapshot": intraday_snapshot.to_dict(),
                    }]
            except (ImportError, Exception):
                # 静默降级：intraday 不可用时不影响主流程
                pass

        # 3. 融合
        if price_layer or trade_flow_layer:
            fusion = naked_k_smart_money_fusion.fuse_dual_evidence(trade_flow_layer, price_layer)
            dual_evidence_fusion = {
                "result": fusion.result.value,
                "direction": fusion.direction,
                "quality": fusion.quality,
                "confidence": fusion.confidence,
                "aligned": fusion.aligned,
                "explanation": fusion.explanation,
                "limitations": list(fusion.limitations),
                "advisory_only": fusion.advisory_only,
                "confirmation_criteria": fusion.confirmation_criteria,
                "invalidation_criteria": fusion.invalidation_criteria,
            }
    except Exception as e:
        # 记录失败但不影响主流程
        try:
            import naked_k_audit
            from pathlib import Path
            audit_path = Path("reports/naked_k_audit.jsonl")
            event = naked_k_audit.build_audit_event(
                "dual_evidence_error",
                payload={
                    "ticker": ticker,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                level="warning"
            )
            naked_k_audit.append_audit_event(audit_path, event)
        except Exception:
            # 如果 audit 本身失败，也不影响主流程
            pass

    review = naked_k_trade.review_previous_call(previous, daily_bar, float(daily_bar["Close"]))
    rationale_parts = [
        f"日线形态：{'、'.join(daily_patterns) if daily_patterns else '无明确信号'}",
        f"周线背景：{'、'.join(weekly_patterns) if weekly_patterns else '无明确信号'}",
        weekly_context,
        f"多周期框架：{naked_k_timeframes.format_timeframe_context(timeframe_context)}",
        f"裸K结构：{naked_k_trade.format_price_action_summary(price_action)}",
        f"市场结构：{naked_k_trade.format_market_structure_summary(market_structure)}",
        f"市场状态：{naked_k_trade.format_market_regime_summary(market_regime)}",
        f"交易剧本：{naked_k_trade.format_trade_setup_summary(trade_setup)}",
        f"关键价格区域：{naked_k_trade.format_price_zones_summary(price_zones)}",
        f"行为上下文：{naked_k_context.format_candle_context_summary(candle_context)}",
        f"量价代理：{_format_smart_money_summary(smart_money_signals)}",
        f"风险计划：{naked_k_trade.format_risk_plan_summary(risk_plan)}",
        f"ATR缓冲：{buffer_ratio * 100:.2f}%",
        "改进：多头/空头都要求先突破信号K极值再触发，减少无确认追价。",
    ]
    if reward_filter_note:
        rationale_parts.append(f"改进：{reward_filter_note}")

    report = InstrumentReport(
        name=name,
        ticker=ticker,
        action=action,
        entry_trigger=entry_trigger,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_per_share=risk_per_share,
        reward_to_risk=reward_to_risk,
        signal_state=signal_state,
        resistance=resistance,
        support=support,
        position_size=position_size,
        rationale="；".join(rationale_parts),
        daily_patterns=daily_patterns,
        weekly_patterns=weekly_patterns,
        weekly_context=weekly_context,
        data_sources={
            "daily": str(daily.attrs.get("source", "unknown")),
            "weekly": str(weekly.attrs.get("source", "unknown")),
            "monthly": str(monthly.attrs.get("source", "unknown")) if monthly is not None else "missing",
        },
        latest_k_dates={
            "daily": daily.index[-1].strftime("%Y-%m-%d"),
            "weekly": weekly.index[-1].strftime("%Y-%m-%d"),
            "monthly": monthly.index[-1].strftime("%Y-%m-%d") if monthly is not None and not monthly.empty else "",
        },
        latest_closes={
            "daily": round(float(daily_bar["Close"]), 2),
            "weekly": round(float(weekly_bar["Close"]), 2),
            "monthly": round(float(monthly["Close"].iloc[-1]), 2) if monthly is not None and not monthly.empty else 0.0,
        },
        review=review,
        improvement="新增裸K增强读线：趋势结构、波动扩张/压缩、回撤深度、量价确认与假突破压力均纳入价格行为上下文。",
        intraday_status=intraday_status,
        price_action=price_action,
        market_structure=market_structure,
        market_regime=market_regime,
        risk_plan=risk_plan,
        trade_setup=trade_setup,
        price_zones=price_zones,
        timeframe_context=timeframe_context,
        candle_context=candle_context,
        smart_money_signals=smart_money_signals,
        dual_evidence_fusion=dual_evidence_fusion,
        price_evidences=price_evidences_list,
        trade_flow_evidences=trade_flow_evidences_list,
    )
    refresh_report_derivatives(report)
    return report


def _format_smart_money_summary(signals: dict[str, Any]) -> str:
    """格式化 OHLCV 量价代理摘要。"""
    if not signals.get("enabled"):
        return "未启用"

    all_signals = [signal for signal in signals.get("signals", []) if not signal.get("stale", False)]
    if not all_signals:
        return signals.get("overall_assessment", "无明显量价代理信号")

    # 显示最高置信度的信号和3个月统计
    top_signals = sorted(
        all_signals,
        key=lambda s: s.get("confidence", 0),
        reverse=True
    )[:2]  # 只显示前2个

    signal_labels = []
    for s in top_signals:
        days_old = s.get("days_old", 0)
        if days_old == 0:
            signal_labels.append(s["label"])
        else:
            signal_labels.append(f"{s['label']}({days_old}天前)")

    # 添加3个月统计
    signal_count_3m = signals.get("signal_count_3m", 0)
    if signal_count_3m > 0:
        summary = f"{signals.get('overall_assessment', '')} ({', '.join(signal_labels)}) | 近3月{signal_count_3m}次"
    else:
        summary = f"{signals.get('overall_assessment', '')} ({', '.join(signal_labels)})"

    return summary
