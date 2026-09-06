from __future__ import annotations

from typing import Any
import math

import naked_k_config


BULLISH_ACTIONS = {"买入", "小仓试错"}
BEARISH_ACTIONS = {"减仓", "回避"}
ACTION_GROSS_CAPS = naked_k_config.DEFAULT_ACTION_GROSS_CAPS


def _direction_for_action(action: str) -> str:
    if action in BULLISH_ACTIONS:
        return "long"
    if action in BEARISH_ACTIONS:
        return "bearish_defensive"
    return "none"


def _risk_level(risk_pct: float) -> str:
    if risk_pct <= 3:
        return "low"
    if risk_pct <= 7:
        return "medium"
    return "high"


def _targets_by_r(direction: str, entry: float, risk_per_share: float) -> dict[str, float]:
    if direction == "long":
        return {
            "1R": round(entry + risk_per_share, 2),
            "2R": round(entry + risk_per_share * 2, 2),
            "3R": round(entry + risk_per_share * 3, 2),
        }
    return {}


def _target_r_multiple(direction: str, entry: float, target: float | None, risk_per_share: float) -> float | None:
    if target is None or risk_per_share <= 0:
        return None
    if direction == "long" and target > entry:
        return round((target - entry) / risk_per_share, 2)
    return None


def _position_size_text(status: str, gross_pct: float, effective_risk_pct: float, risk_pct: float) -> str:
    if status == "blocked":
        return "0%（风险保护触发，暂停新仓）"
    if gross_pct <= 0:
        return "0%（无新仓计划）"
    return f"最高约{gross_pct:.1f}%仓位（按{effective_risk_pct:g}%账户风险，单股风险{risk_pct:.1f}%）"


def build_risk_plan(
    action: str,
    entry_trigger: float,
    stop_loss: float,
    target_price: float | None,
    account_risk_pct: float = 1.0,
    current_drawdown_pct: float = 0.0,
    max_drawdown_pct: float = 8.0,
    consecutive_losses: int = 0,
    config: naked_k_config.RiskConfig | None = None,
) -> dict[str, Any]:
    naked_k_config.number(entry_trigger, "entry_trigger", 1e-12, float("inf"))
    naked_k_config.number(stop_loss, "stop_loss", 1e-12, float("inf"))
    naked_k_config.number(account_risk_pct, "account_risk_pct")
    naked_k_config.number(current_drawdown_pct, "current_drawdown_pct")
    naked_k_config.number(max_drawdown_pct, "max_drawdown_pct", 0.01)
    if type(consecutive_losses) is not int or consecutive_losses < 0:
        raise ValueError("consecutive_losses must be a nonnegative integer")
    if action in BULLISH_ACTIONS and stop_loss >= entry_trigger:
        raise ValueError("long stop must be below entry")
    if config is not None:
        if account_risk_pct == 1.0:
            account_risk_pct = config.account_risk_pct
        if max_drawdown_pct == 8.0:
            max_drawdown_pct = config.max_drawdown_pct
        consecutive_loss_limit = config.consecutive_loss_limit
        consecutive_loss_risk_multiplier = config.consecutive_loss_risk_multiplier
        action_gross_caps = config.action_gross_caps
    else:
        consecutive_loss_limit = 3
        consecutive_loss_risk_multiplier = 0.5
        action_gross_caps = ACTION_GROSS_CAPS

    direction = _direction_for_action(action)
    risk_per_share = abs(entry_trigger - stop_loss)
    risk_pct = risk_per_share / entry_trigger * 100
    max_gross_pct = float(action_gross_caps.get(action, 0.0))
    guardrails: list[str] = []
    status = "active" if direction == "long" and max_gross_pct > 0 else "flat"
    effective_account_risk_pct = float(account_risk_pct)

    if current_drawdown_pct >= max_drawdown_pct:
        status = "blocked"
        effective_account_risk_pct = 0.0
        guardrails.append("最大回撤保护")
    elif consecutive_losses >= consecutive_loss_limit and status == "active":
        status = "reduced"
        effective_account_risk_pct = account_risk_pct * consecutive_loss_risk_multiplier
        guardrails.append("连续亏损降风险")

    if status in {"blocked", "flat"} or risk_pct <= 0:
        suggested_gross_pct = 0.0
        # flat 计划不分配风险预算（虽然有止损位，但未开仓）
        if status == "flat":
            effective_account_risk_pct = 0.0
    else:
        risk_budget_cap = effective_account_risk_pct / risk_pct * 100
        suggested_gross_pct = math.floor(min(max_gross_pct, risk_budget_cap) * 10 + 1e-9) / 10

    targets = _targets_by_r(direction, float(entry_trigger), risk_per_share)
    return {
        "status": status,
        "direction": direction,
        "entry": float(entry_trigger),
        "stop": float(stop_loss),
        "target": float(target_price) if target_price is not None else None,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "risk_level": _risk_level(risk_pct),
        "base_account_risk_pct": float(account_risk_pct),
        "effective_account_risk_pct": effective_account_risk_pct,
        "current_drawdown_pct": float(current_drawdown_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "consecutive_losses": int(consecutive_losses),
        "consecutive_loss_limit": int(consecutive_loss_limit),
        "max_gross_pct": max_gross_pct,
        "suggested_gross_pct": suggested_gross_pct,
        "targets_by_r": targets,
        "target_r_multiple": _target_r_multiple(direction, float(entry_trigger), target_price, risk_per_share),
        "guardrails": guardrails,
        "position_size": _position_size_text(status, suggested_gross_pct, effective_account_risk_pct, risk_pct),
    }
