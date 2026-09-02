from __future__ import annotations

import copy
import ipaddress
import math
import re
import unicodedata
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

import naked_k_config
import naked_k_news_llm
import naked_k_portfolio
import naked_k_risk
import naked_k_trade


TECHNICAL_SNAPSHOT_FIELDS = (
    "action",
    "signal_state",
    "entry_trigger",
    "stop_loss",
    "target_price",
    "risk_per_share",
    "reward_to_risk",
    "position_size",
    "resistance",
    "support",
    "rationale",
    "risk_plan",
    "intraday_status",
)

ACTION_SIDE_MAP = {
    "买入": "long",
    "小仓试错": "long",
    "观望": "neutral",
    "减仓": "bearish_defensive",
    "回避": "bearish_defensive",
}

_GROUNDING_OVERRIDE_CODE = "news_action_change_grounding_required"
_CORROBORATION_OVERRIDE_CODE = "news_upgrade_independent_corroboration_required"
_QUARANTINE_OVERRIDE_CODE = "news_instruction_quarantine_action_change_blocked"
_INVALID_CANDIDATE_RISK_CODE = "news_candidate_executable_risk_invalid"
_GROUNDING_OVERRIDE_REASON = "消息动作变更缺少有效的结构化证据，已保留技术动作"
_CORROBORATION_OVERRIDE_REASON = "消息增仓建议未获至少两个独立来源交叉佐证，已保留技术动作"
_QUARANTINE_OVERRIDE_REASON = "消息输入含已隔离的指令式内容，禁止据此改变技术动作"
_INVALID_CANDIDATE_RISK_REASON = "消息动作生成的候选风险计划含无效数值，已保留技术动作"
_EXECUTABLE_RISK_MEASURES = (
    "suggested_gross_pct",
    "effective_account_risk_pct",
    "max_gross_pct",
)
_MULTI_LABEL_PUBLIC_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.cn", "net.cn", "org.cn", "gov.cn",
    "com.hk", "net.hk", "org.hk",
    "co.jp", "ne.jp", "or.jp",
    "co.kr", "or.kr", "go.kr",
    "com.au", "net.au", "org.au",
    "co.nz", "org.nz",
    "com.sg", "com.tw", "com.br", "com.mx", "co.in",
}
_PUBLISHER_ALIASES = {
    "reuters": "reuters",
    "thomsonreuters": "reuters",
    "路透": "reuters",
    "路透社": "reuters",
    "bloomberg": "bloomberg",
    "bloombergnews": "bloomberg",
    "彭博": "bloomberg",
    "彭博社": "bloomberg",
    "associatedpress": "associated-press",
    "ap": "associated-press",
    "美联社": "associated-press",
    "xinhua": "xinhua",
    "xinhuanews": "xinhua",
    "新华社": "xinhua",
}


def snapshot_technical_conclusion(report: Any) -> dict[str, Any]:
    return copy.deepcopy(
        {field: getattr(report, field) for field in TECHNICAL_SNAPSHOT_FIELDS}
    )


def build_risk_context(
    technical_snapshot: dict[str, Any],
    trading_config: naked_k_config.TradingConfig | None = None,
) -> dict[str, Any]:
    active_config = trading_config or naked_k_config.TradingConfig()
    return {
        "technical_risk_plan": copy.deepcopy(technical_snapshot.get("risk_plan", {})),
        "risk_limits": asdict(active_config.risk),
        "portfolio_limits": asdict(active_config.portfolio),
    }


def side_for_action(action: str) -> str:
    try:
        return ACTION_SIDE_MAP[action]
    except KeyError as exc:
        raise ValueError(f"unsupported synthesis action: {action}") from exc


def _normalized_confidence(value: Any) -> float:
    try:
        confidence = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return 0.0
    if not confidence.is_finite():
        return 0.0
    return float(min(Decimal(100), max(Decimal(0), confidence)))


def _stored_technical_snapshot(report: Any) -> dict[str, Any]:
    stored = getattr(report, "technical_conclusion", None)
    if isinstance(stored, dict) and all(field in stored for field in TECHNICAL_SNAPSHOT_FIELDS):
        return copy.deepcopy(stored)
    return snapshot_technical_conclusion(report)


def _normalized_publisher(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    words = " ".join(re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", normalized).split())
    alias_key = words.replace(" ", "").replace("_", "")
    if (
        alias_key.startswith("reuters")
        or alias_key.startswith("thomsonreuters")
        or alias_key.startswith("路透")
    ):
        return "reuters"
    return _PUBLISHER_ALIASES.get(alias_key, words)


def _source_domain(value: Any) -> str:
    try:
        hostname = urlsplit(str(value or "")).hostname or ""
    except ValueError:
        return ""
    normalized = hostname.rstrip(".").casefold()
    if not normalized:
        return ""
    try:
        return str(ipaddress.ip_address(normalized))
    except ValueError:
        pass
    labels = [label for label in normalized.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    suffix = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix in _MULTI_LABEL_PUBLIC_SUFFIXES else suffix


def _action_exposure_cap(
    action: str,
    config: naked_k_config.TradingConfig,
) -> float:
    if action in {"回避", "观望"}:
        return 0.0
    try:
        return max(0.0, float(config.risk.action_gross_caps.get(action, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _finite_nonnegative_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def _executable_risk_measures(snapshot: dict[str, Any]) -> dict[str, float]:
    risk_plan = snapshot.get("risk_plan")
    if not isinstance(risk_plan, dict):
        risk_plan = {}
    return {
        field: _finite_nonnegative_number(risk_plan.get(field))
        for field in _EXECUTABLE_RISK_MEASURES
    }


def _candidate_executable_risk_measures(
    snapshot: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    risk_plan = snapshot.get("risk_plan")
    if not isinstance(risk_plan, dict):
        risk_plan = {}
    measures: dict[str, float] = {}
    invalid_fields: list[str] = []
    for field in _EXECUTABLE_RISK_MEASURES:
        value = risk_plan.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
            invalid_fields.append(field)
        else:
            if not math.isfinite(number) or number < 0.0:
                number = 0.0
                invalid_fields.append(field)
        measures[field] = number
    return measures, invalid_fields


def _instruction_quarantine_state(
    news_analysis: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    quarantine = news_analysis.get("quarantine")
    recorded_count = 0
    recorded_ids: list[str] = []
    if isinstance(quarantine, dict):
        recorded_count = int(_finite_nonnegative_number(quarantine.get("count")))
        raw_ids = quarantine.get("evidence_ids")
        if isinstance(raw_ids, list):
            recorded_ids = [
                item
                for item in raw_ids
                if naked_k_news_llm.is_safe_quarantine_evidence_id(item)
            ]

    scanned_ids: list[str] = []
    scanned_count = 0
    for item in items:
        if not naked_k_news_llm.news_item_contains_instruction(item):
            continue
        scanned_count += 1
        evidence_id = item.get("id")
        if naked_k_news_llm.is_safe_quarantine_evidence_id(evidence_id):
            scanned_ids.append(evidence_id)
    return max(recorded_count, scanned_count), sorted(set(recorded_ids + scanned_ids))


def _evidence_safety_gate(
    report: Any,
    deliberation: dict[str, Any],
    *,
    technical_action: str,
    config: naked_k_config.TradingConfig | None = None,
    candidate: dict[str, Any] | None = None,
    evaluate_exposure: bool = True,
) -> dict[str, Any]:
    active_config = config or naked_k_config.TradingConfig()
    model_action = str(deliberation.get("model_action", ""))
    action_changed = model_action != technical_action
    evidence_ids_value = deliberation.get("evidence_ids")
    evidence_ids = (
        list(dict.fromkeys(
            item
            for item in evidence_ids_value
            if isinstance(item, str) and item
        ))
        if isinstance(evidence_ids_value, list)
        else []
    )
    claims_value = deliberation.get("evidence_claims")

    news_analysis = getattr(report, "news_analysis", {})
    collection = (
        news_analysis.get("collection")
        if isinstance(news_analysis, dict)
        else {}
    )
    items = collection.get("items") if isinstance(collection, dict) else []
    items = items if isinstance(items, list) else []
    items_by_id = (
        {
            item["id"]: item
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if isinstance(items, list)
        else {}
    )
    cited_items = [items_by_id[item] for item in evidence_ids if item in items_by_id]
    all_ids_resolve = len(cited_items) == len(evidence_ids)
    try:
        validated_claims = naked_k_news_llm.validate_evidence_claims(
            claims_value,
            items=items,
            evidence_ids=evidence_ids,
        )
    except naked_k_news_llm.NewsValidationError:
        validated_claims = []
    structured_grounding_passed = (
        bool(evidence_ids)
        and bool(validated_claims)
        and all_ids_resolve
    )

    publishers = sorted({
        publisher
        for item in cited_items
        if (publisher := _normalized_publisher(item.get("publisher")))
    })
    domains = sorted({
        domain
        for item in cited_items
        if (domain := _source_domain(item.get("url")))
    })
    proposition_sources: dict[str, dict[str, set[str]]] = {}
    for claim in validated_claims:
        fingerprint = naked_k_news_llm.material_proposition_fingerprint(
            str(claim.get("claim", ""))
        )
        source = items_by_id.get(str(claim.get("evidence_id", "")))
        if not fingerprint or not isinstance(source, dict):
            continue
        group = proposition_sources.setdefault(
            fingerprint,
            {"publishers": set(), "domains": set(), "evidence_ids": set()},
        )
        publisher = _normalized_publisher(source.get("publisher"))
        domain = _source_domain(source.get("url"))
        evidence_id = source.get("id")
        if publisher:
            group["publishers"].add(publisher)
        if domain:
            group["domains"].add(domain)
        if isinstance(evidence_id, str) and evidence_id:
            group["evidence_ids"].add(evidence_id)
    corroborated_fingerprints = sorted(
        fingerprint
        for fingerprint, sources in proposition_sources.items()
        if (
            len(sources["publishers"]) >= 2
            and len(sources["domains"]) >= 2
            and len(sources["evidence_ids"]) >= 2
        )
    )
    independent_corroboration = bool(corroborated_fingerprints)

    technical_snapshot = _stored_technical_snapshot(report)
    baseline_measures = _executable_risk_measures(technical_snapshot)
    invalid_candidate_measures: list[str] = []
    if isinstance(candidate, dict):
        candidate_measures, invalid_candidate_measures = (
            _candidate_executable_risk_measures(candidate)
        )
    else:
        candidate_measures = copy.deepcopy(baseline_measures)
    increased_measures: list[str] = []
    if evaluate_exposure:
        increased_measures = [
            field
            for field in _EXECUTABLE_RISK_MEASURES
            if candidate_measures[field] > baseline_measures[field] + 1e-9
        ]
        candidate_action = (
            str(candidate.get("action", model_action))
            if isinstance(candidate, dict)
            else model_action
        )
        if (
            _action_exposure_cap(candidate_action, active_config)
            > _action_exposure_cap(technical_action, active_config) + 1e-9
        ):
            increased_measures.append("configured_action_gross_cap")
    exposure_increase = bool(increased_measures)
    quarantine_count, quarantine_ids = _instruction_quarantine_state(
        news_analysis if isinstance(news_analysis, dict) else {},
        items,
    )

    reason_code = ""
    reason = ""
    passed = True
    if action_changed and not structured_grounding_passed:
        passed = False
        reason_code = _GROUNDING_OVERRIDE_CODE
        reason = _GROUNDING_OVERRIDE_REASON
    elif action_changed and quarantine_count:
        passed = False
        reason_code = _QUARANTINE_OVERRIDE_CODE
        reason = _QUARANTINE_OVERRIDE_REASON
    elif invalid_candidate_measures:
        passed = False
        reason_code = _INVALID_CANDIDATE_RISK_CODE
        reason = _INVALID_CANDIDATE_RISK_REASON
    elif exposure_increase and not independent_corroboration:
        passed = False
        reason_code = _CORROBORATION_OVERRIDE_CODE
        reason = _CORROBORATION_OVERRIDE_REASON
    return {
        "required": action_changed or exposure_increase or bool(invalid_candidate_measures),
        "exposure_increase": exposure_increase,
        "passed": passed,
        "reason_code": reason_code,
        "reason": reason,
        "cited_evidence_ids": evidence_ids,
        "independent_publisher_count": len(publishers),
        "independent_domain_count": len(domains),
        "required_independent_sources": 2 if exposure_increase else 0,
        "corroborated_proposition_count": len(corroborated_fingerprints),
        "corroborated_proposition_fingerprints": corroborated_fingerprints,
        "baseline_executable_risk": baseline_measures,
        "candidate_executable_risk": candidate_measures,
        "invalid_candidate_executable_measures": invalid_candidate_measures,
        "increased_executable_measures": list(dict.fromkeys(increased_measures)),
        "quarantined_evidence_count": quarantine_count,
        "quarantined_evidence_ids": quarantine_ids,
    }


def _deterministic_prices(
    technical_snapshot: dict[str, Any],
    daily: pd.DataFrame,
    action: str,
) -> tuple[float, float]:
    final_side = side_for_action(action)
    technical_side = side_for_action(str(technical_snapshot["action"]))

    if action == "观望":
        price_side = "bullish"
    elif final_side == technical_side:
        return (
            float(technical_snapshot["entry_trigger"]),
            float(technical_snapshot["stop_loss"]),
        )
    elif final_side == "long":
        price_side = "bullish"
    else:
        price_side = "bearish"

    buffer_ratio = naked_k_trade.build_volatility_buffer_ratio(daily)
    latest_bar = daily.iloc[-1]
    return (
        naked_k_trade.build_breakout_trigger(
            latest_bar,
            price_side,
            buffer_ratio=buffer_ratio,
        ),
        naked_k_trade.build_invalidation_level(
            latest_bar,
            price_side,
            buffer_ratio=buffer_ratio,
        ),
    )


def _build_candidate(
    technical_snapshot: dict[str, Any],
    daily: pd.DataFrame,
    action: str,
    *,
    reason: str,
    intraday: pd.DataFrame | None,
    config: naked_k_config.TradingConfig | None,
) -> dict[str, Any]:
    side_for_action(action)
    entry_trigger, stop_loss = _deterministic_prices(technical_snapshot, daily, action)
    target_price, risk_per_share, reward_to_risk = naked_k_trade.build_trade_metrics(
        action,
        entry_trigger,
        stop_loss,
        float(technical_snapshot["resistance"]),
        float(technical_snapshot["support"]),
    )
    protected_action, _, _, reward_filter_note = naked_k_trade.downgrade_low_reward_setup(
        action,
        target_price,
        reward_to_risk,
    )

    technical_risk = technical_snapshot.get("risk_plan")
    if not isinstance(technical_risk, dict):
        technical_risk = {}
    risk_plan = naked_k_risk.build_risk_plan(
        action=action,
        entry_trigger=entry_trigger,
        stop_loss=stop_loss,
        target_price=target_price,
        current_drawdown_pct=float(technical_risk.get("current_drawdown_pct", 0.0)),
        consecutive_losses=int(technical_risk.get("consecutive_losses", 0)),
        config=config.risk if config is not None else None,
        defensive_residual=action == "减仓",
    )

    protection_reasons: list[str] = []
    if reward_filter_note:
        protection_reasons.append(reward_filter_note)
    if action in naked_k_trade.BULLISH_ACTIONS and risk_plan.get("status") == "blocked":
        protected_action = "观望"
        guardrails = risk_plan.get("guardrails")
        if isinstance(guardrails, list) and guardrails:
            protection_reasons.extend(str(item) for item in guardrails)
        else:
            protection_reasons.append("风险保护触发")

    position_guidance = naked_k_trade.build_position_guidance(
        action,
        entry_trigger,
        stop_loss,
    )
    position_size = (
        str(risk_plan["position_size"])
        if action in naked_k_trade.BULLISH_ACTIONS
        else position_guidance
    )
    signal_state = naked_k_trade.build_signal_state(action)

    execution_side = side_for_action(action)
    if execution_side == "bearish_defensive":
        risk_plan["engine_direction"] = risk_plan.get("direction")
        risk_plan["direction"] = "bearish_defensive"
        risk_plan["position_intent"] = "reduce_or_avoid_long_exposure"
        signal_state = "planned_defensive"

    if action in {"观望", "回避"}:
        risk_plan["status"] = "flat"
        risk_plan["suggested_gross_pct"] = 0.0
        risk_plan["effective_account_risk_pct"] = 0.0
        risk_plan["position_size"] = "0%（无新仓计划）"
    if action == "观望":
        target_price = None
        reward_to_risk = None
        position_size = str(risk_plan["position_size"])
        signal_state = "watching"

    base_rationale = str(technical_snapshot.get("rationale", ""))
    rationale = f"{base_rationale}；综合结论：{reason}" if reason else base_rationale
    intraday_status = naked_k_trade.build_intraday_status(
        intraday,
        action,
        entry_trigger,
        stop_loss,
    )
    return {
        "action": action,
        "signal_state": signal_state,
        "entry_trigger": entry_trigger,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_per_share": risk_per_share,
        "reward_to_risk": reward_to_risk,
        "position_size": position_size,
        "resistance": technical_snapshot["resistance"],
        "support": technical_snapshot["support"],
        "rationale": rationale,
        "risk_plan": risk_plan,
        "intraday_status": intraday_status,
        "protected_action": protected_action,
        "protection_reason": "；".join(dict.fromkeys(protection_reasons)),
    }


def _apply_candidate(report: Any, candidate: dict[str, Any]) -> None:
    for field in TECHNICAL_SNAPSHOT_FIELDS:
        setattr(report, field, copy.deepcopy(candidate[field]))


def _synchronized_candidate(
    technical_snapshot: dict[str, Any],
    daily: pd.DataFrame,
    action: str,
    *,
    reason: str,
    intraday: pd.DataFrame | None,
    config: naked_k_config.TradingConfig | None,
) -> tuple[dict[str, Any], str]:
    candidate = _build_candidate(
        technical_snapshot,
        daily,
        action,
        reason=reason,
        intraday=intraday,
        config=config,
    )
    protected_action = str(candidate["protected_action"])
    protection_reason = str(candidate["protection_reason"])
    if protected_action != action:
        protected_reason = "；".join(part for part in (reason, protection_reason) if part)
        candidate = _build_candidate(
            technical_snapshot,
            daily,
            protected_action,
            reason=protected_reason,
            intraday=intraday,
            config=config,
        )
    return candidate, protection_reason


def _clamp_defensive_residual_exposure(
    candidate: dict[str, Any],
    technical_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("action") != "减仓":
        return candidate
    risk_plan = candidate.get("risk_plan")
    technical_risk = technical_snapshot.get("risk_plan")
    if not isinstance(risk_plan, dict):
        return candidate
    if not isinstance(technical_risk, dict):
        technical_risk = {}

    baseline_gross = _finite_nonnegative_number(
        technical_risk.get("suggested_gross_pct")
    )
    baseline_account_risk = _finite_nonnegative_number(
        technical_risk.get("effective_account_risk_pct")
    )
    clamped_gross = min(
        _finite_nonnegative_number(risk_plan.get("suggested_gross_pct")),
        baseline_gross,
    )
    clamped_account_risk = min(
        _finite_nonnegative_number(risk_plan.get("effective_account_risk_pct")),
        baseline_account_risk,
    )
    if clamped_gross <= 0 or clamped_account_risk <= 0:
        clamped_gross = 0.0
        clamped_account_risk = 0.0
    risk_plan["suggested_gross_pct"] = round(clamped_gross, 1)
    risk_plan["effective_account_risk_pct"] = clamped_account_risk
    risk_plan["max_gross_pct"] = min(
        _finite_nonnegative_number(risk_plan.get("max_gross_pct")),
        clamped_gross,
    )
    if "base_account_risk_pct" in risk_plan:
        baseline_base_risk = _finite_nonnegative_number(
            technical_risk.get("base_account_risk_pct", baseline_account_risk)
        )
        risk_plan["base_account_risk_pct"] = min(
            _finite_nonnegative_number(risk_plan.get("base_account_risk_pct")),
            baseline_base_risk,
        )
    guardrails = risk_plan.get("guardrails")
    if not isinstance(guardrails, list):
        guardrails = []
    risk_plan["guardrails"] = list(dict.fromkeys(
        [*guardrails, "减仓残余敞口不超过技术基线"]
    ))
    if clamped_gross <= 0:
        risk_plan["status"] = "flat"
        position_size = "0%（无剩余多头敞口）"
    else:
        position_size = (
            f"降至{clamped_gross:.1f}%以内"
            f"（账户风险不高于{clamped_account_risk:g}%；仅处理已有多头，不新建仓）"
        )
    risk_plan["position_size"] = position_size
    candidate["position_size"] = position_size
    return candidate


def synchronize_final_action(
    report: Any,
    daily: pd.DataFrame,
    final_action: str,
    *,
    reason: str,
    intraday: pd.DataFrame | None = None,
    config: naked_k_config.TradingConfig | None = None,
) -> None:
    technical_snapshot = _stored_technical_snapshot(report)
    candidate, _ = _synchronized_candidate(
        technical_snapshot,
        daily,
        final_action,
        reason=reason,
        intraday=intraday,
        config=config,
    )
    candidate = _clamp_defensive_residual_exposure(candidate, technical_snapshot)
    _apply_candidate(report, candidate)


def _combined_conclusion(
    deliberation: dict[str, Any],
    *,
    status: str,
    final_action: str,
    risk_override_reason: str,
    override_reason_code: str,
    evidence_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "technical_view": copy.deepcopy(deliberation["technical_view"]),
        "news_view": copy.deepcopy(deliberation["news_view"]),
        "conflict_analysis": str(deliberation["conflict_analysis"]),
        "model_action": str(deliberation["model_action"]),
        "final_action": final_action,
        "confidence": deliberation["confidence"],
        "decision_reasons": copy.deepcopy(deliberation["decision_reasons"]),
        "risk_flags": copy.deepcopy(deliberation["risk_flags"]),
        "evidence_ids": copy.deepcopy(deliberation["evidence_ids"]),
        "evidence_claims": copy.deepcopy(deliberation.get("evidence_claims", [])),
        "execution_note": str(deliberation["execution_note"]),
        "execution_side": side_for_action(final_action),
        "risk_override_reason": risk_override_reason,
        "override_reason_code": override_reason_code,
        "evidence_gate": copy.deepcopy(evidence_gate),
        "price_plan_source": "deterministic_naked_k",
    }


def apply_deliberation(
    report: Any,
    daily: pd.DataFrame,
    deliberation: dict[str, Any],
    *,
    intraday: pd.DataFrame | None = None,
    config: naked_k_config.TradingConfig | None = None,
) -> dict[str, Any]:
    technical_snapshot = _stored_technical_snapshot(report)
    model_action = str(deliberation["model_action"])
    decision_reason = "；".join(str(item) for item in deliberation["decision_reasons"])
    evidence_gate = _evidence_safety_gate(
        report,
        deliberation,
        technical_action=str(technical_snapshot["action"]),
        config=config,
        evaluate_exposure=False,
    )
    if not evidence_gate["passed"]:
        # The gate outcome, reason, and reason_code are carried on the returned
        # combined conclusion and recorded by the caller's audit logger; no
        # separate stderr channel is needed.
        for field in TECHNICAL_SNAPSHOT_FIELDS:
            setattr(report, field, copy.deepcopy(technical_snapshot[field]))
        combined = _combined_conclusion(
            deliberation,
            status="ok",
            final_action=str(technical_snapshot["action"]),
            risk_override_reason=str(evidence_gate["reason"]),
            override_reason_code=str(evidence_gate["reason_code"]),
            evidence_gate=evidence_gate,
        )
        report.combined_conclusion = combined
        return combined

    try:
        candidate, protection_reason = _synchronized_candidate(
            technical_snapshot,
            daily,
            model_action,
            reason=decision_reason,
            intraday=intraday,
            config=config,
        )
        evidence_gate = _evidence_safety_gate(
            report,
            deliberation,
            technical_action=str(technical_snapshot["action"]),
            config=config,
            candidate=candidate,
        )
        if not evidence_gate["passed"]:
            for field in TECHNICAL_SNAPSHOT_FIELDS:
                setattr(report, field, copy.deepcopy(technical_snapshot[field]))
            combined = _combined_conclusion(
                deliberation,
                status="ok",
                final_action=str(technical_snapshot["action"]),
                risk_override_reason=str(evidence_gate["reason"]),
                override_reason_code=str(evidence_gate["reason_code"]),
                evidence_gate=evidence_gate,
            )
            report.combined_conclusion = combined
            return combined
        candidate = _clamp_defensive_residual_exposure(
            candidate,
            technical_snapshot,
        )
        final_action = str(candidate["action"])
        _apply_candidate(report, candidate)
        override_reasons = []
        if protection_reason and final_action != model_action:
            override_reasons.append(protection_reason)
        risk_override_reason = "；".join(dict.fromkeys(override_reasons))
        override_reason_code = str(evidence_gate["reason_code"])
        combined = _combined_conclusion(
            deliberation,
            status="ok",
            final_action=final_action,
            risk_override_reason=risk_override_reason,
            override_reason_code=override_reason_code,
            evidence_gate=evidence_gate,
        )
    except Exception as exc:
        error_detail = f"{type(exc).__name__}: {exc}"
        for field in TECHNICAL_SNAPSHOT_FIELDS:
            setattr(report, field, copy.deepcopy(technical_snapshot[field]))
        final_action = str(technical_snapshot["action"])
        fallback_reason = (
            "确定性价格计划重建失败，已安全回退技术结论："
            f"{error_detail}"
        )
        combined = _combined_conclusion(
            deliberation,
            status="technical_fallback",
            final_action=final_action,
            risk_override_reason=fallback_reason,
            override_reason_code="deterministic_synthesis_failure",
            evidence_gate=evidence_gate,
        )
        # Deliberately NOT storing the raw exception message. It can embed prompt
        # text or credentials, this dict is serialized into the journal, and this
        # module has no news config with which to redact. The class name already
        # reaches the audit log via risk_override_reason and error_type.

    report.combined_conclusion = combined
    return combined


def apply_portfolio_guardrails(
    reports: list[Any],
    daily_by_ticker: dict[str, pd.DataFrame],
    *,
    intraday_by_ticker: dict[str, pd.DataFrame | None] | None = None,
    config: naked_k_config.TradingConfig | None = None,
) -> dict[str, Any]:
    """Return final exposure and deterministic override metadata."""
    active_config = config or naked_k_config.TradingConfig()
    intraday_frames = intraday_by_ticker or {}
    overridden_report_ids: set[int] = set()
    overrides: list[dict[str, Any]] = []
    exposure = naked_k_portfolio.evaluate_portfolio_exposure(
        reports,
        active_config.portfolio,
    )

    while exposure["status"] == "over_limit":
        candidates: list[tuple[float, float, str, Any, dict[str, Any]]] = []
        for report in reports:
            combined = getattr(report, "combined_conclusion", None)
            if not isinstance(combined, dict) or combined.get("status") != "ok":
                continue
            if id(report) in overridden_report_ids:
                continue
            action = str(getattr(report, "action", ""))
            if action not in {"买入", "小仓试错", "减仓"}:
                continue
            risk_plan = getattr(report, "risk_plan", {})
            if not isinstance(risk_plan, dict):
                continue
            gross_pct = float(risk_plan.get("suggested_gross_pct", 0.0) or 0.0)
            account_risk_pct = float(
                risk_plan.get("effective_account_risk_pct", 0.0) or 0.0
            )
            if gross_pct <= 0.0 and account_risk_pct <= 0.0:
                continue
            ticker = str(getattr(report, "ticker", ""))
            confidence = _normalized_confidence(combined.get("confidence"))
            candidates.append((confidence, -gross_pct, ticker, report, combined))

        if not candidates:
            break

        _, _, ticker, report, combined = min(candidates, key=lambda item: item[:3])
        prior_final_action = str(getattr(report, "action", ""))
        protected_final_action = "回避" if prior_final_action == "减仓" else "观望"
        guardrail_reason = "组合风险保护：" + "；".join(
            str(reason) for reason in exposure["guardrails"]
        )
        synchronize_final_action(
            report,
            daily_by_ticker[ticker],
            protected_final_action,
            reason=guardrail_reason,
            intraday=intraday_frames.get(ticker),
            config=active_config,
        )
        combined["final_action"] = protected_final_action
        combined["execution_side"] = side_for_action(protected_final_action)
        combined["risk_override_reason"] = guardrail_reason
        overridden_report_ids.add(id(report))
        overrides.append(
            {
                "ticker": ticker,
                "model_action": str(combined.get("model_action", "")),
                "prior_final_action": prior_final_action,
                "protected_final_action": protected_final_action,
                "guardrail_reason": guardrail_reason,
            }
        )
        exposure = naked_k_portfolio.evaluate_portfolio_exposure(
            reports,
            active_config.portfolio,
        )

    return {
        **exposure,
        "overrides": overrides,
        "unresolved_guardrails": (
            list(exposure["guardrails"])
            if exposure["status"] == "over_limit"
            else []
        ),
    }
