"""Validated risk limits; indicator rules are frozen in naked_k_trend."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


DEFAULT_ACTION_GROSS_CAPS = {'买入': 30.0, '小仓试错': 15.0, '减仓': 0.0, '回避': 0.0, '观望': 0.0}


def number(value: Any, name: str, minimum: float = 0, maximum: float = 100) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be finite and in [{minimum}, {maximum}]')
    return float(value)


@dataclass(frozen=True)
class RiskConfig:
    account_risk_pct: float = 1.0
    max_drawdown_pct: float = 8.0
    consecutive_loss_limit: int = 3
    consecutive_loss_risk_multiplier: float = 0.5
    action_gross_caps: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_GROSS_CAPS))

    def __post_init__(self):
        number(self.account_risk_pct, 'account_risk_pct')
        number(self.max_drawdown_pct, 'max_drawdown_pct', 0.01)
        if type(self.consecutive_loss_limit) is not int or self.consecutive_loss_limit < 1:
            raise ValueError('consecutive_loss_limit must be a positive integer')
        number(self.consecutive_loss_risk_multiplier, 'consecutive_loss_risk_multiplier', 0, 1)
        if not isinstance(self.action_gross_caps, dict) or set(self.action_gross_caps) - set(DEFAULT_ACTION_GROSS_CAPS):
            raise ValueError('invalid action_gross_caps')
        for action, cap in self.action_gross_caps.items():
            number(cap, f'action_gross_caps.{action}')


@dataclass(frozen=True)
class PortfolioConfig:
    max_total_gross_pct: float = 80.0
    max_direction_gross_pct: float = 60.0
    max_market_gross_pct: float = 40.0
    max_single_name_gross_pct: float = 30.0
    max_total_account_risk_pct: float = 3.0

    def __post_init__(self):
        for item in fields(self):
            number(getattr(self, item.name), item.name)


@dataclass(frozen=True)
class TradingConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)


def build_trading_config(payload: dict[str, Any] | None = None) -> TradingConfig:
    data = {} if payload is None else payload
    if not isinstance(data, dict):
        raise ValueError('config must be a JSON object')
    if 'smart_money' in data:
        raise ValueError('smart_money 已移除；请按 config.example.json 迁移配置')
    if set(data) - {'risk', 'portfolio'}:
        raise ValueError('unknown config fields')
    sections = []
    for key, cls in [('risk', RiskConfig), ('portfolio', PortfolioConfig)]:
        values = data.get(key, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f'invalid {key} configuration fields')
        values = dict(values)
        if key == 'risk' and 'action_gross_caps' in values:
            caps = values['action_gross_caps']
            if not isinstance(caps, dict):
                raise ValueError('action_gross_caps must be an object')
            values['action_gross_caps'] = {**DEFAULT_ACTION_GROSS_CAPS, **caps}
        sections.append(cls(**values))
    return TradingConfig(*sections)


def load_trading_config(path: str | Path | None = None) -> TradingConfig:
    return build_trading_config(json.loads(Path(path).read_text(encoding='utf-8')) if path else None)
