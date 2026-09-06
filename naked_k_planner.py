"""Turn the shared trend snapshot into a conditional, account-aware plan."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import naked_k_config as settings
import naked_k_portfolio as portfolio
import naked_k_risk
import naked_k_trend


@dataclass
class InstrumentReport:
    name: str
    ticker: str
    action: str
    signal_state: str
    trend: dict[str, Any]
    risk_plan: dict[str, Any]
    account: dict[str, Any]
    management: dict[str, Any]
    entry_trigger: float | None
    stop_loss: float | None
    target_price: float | None
    risk_per_share: float | None
    reward_to_risk: float | None
    support: float | None
    resistance: float | None
    position_size: str
    data_sources: dict[str, str]
    latest_k_dates: dict[str, str]
    latest_closes: dict[str, float]
    review: dict[str, Any]
    schema_version: str = naked_k_trend.RULE_VERSION
    news: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)


def _flat_risk(reason: str) -> dict[str, Any]:
    return {'status': 'flat', 'direction': 'none', 'suggested_gross_pct': 0.0,
            'effective_account_risk_pct': 0.0, 'planned_account_risk_pct': 0.0,
            'risk_pct': None, 'guardrails': [reason] if reason else [],
            'position_size': '0%（无新仓计划）'}


def build_trade_plan(name: str, ticker: str, daily: pd.DataFrame,
                     weekly: pd.DataFrame | None = None, previous: dict | None = None, *,
                     config: settings.TradingConfig | None = None,
                     account_state: dict | None = None) -> InstrumentReport:
    config = config or settings.TradingConfig()
    ticker = ticker.strip().upper()
    trend = naked_k_trend.analyze_trend(daily)
    as_of = str(trend['as_of'])[:10]
    account = portfolio.assess_account(account_state, as_of, config)
    known = account['status'] == 'known'
    holding = next((x for x in account['positions'] if x['ticker'] == ticker.upper() and x['gross_pct'] > 0), None) if known else None
    stop = trend['stop_loss']
    prior_stop = holding.get('stop_loss') if holding else None
    suggested_stop = max(x for x in (stop, prior_stop) if x is not None) if any(x is not None for x in (stop, prior_stop)) else None
    stop_breached = bool(holding and prior_stop is not None and trend['close'] <= prior_stop)
    needs_exit = bool(holding and (trend['exit_signal'] or stop_breached))
    management = {
        'holding_status': ('held' if holding else 'flat') if known else 'unknown',
        'exit_next_open': needs_exit,
        'stop_breached': stop_breached,
        'conditional_exit': bool(trend['exit_signal']),
        'previous_stop': prior_stop,
        'suggested_stop': suggested_stop if holding and not stop_breached else None,
        'note': '若已有多头：收盘跌破EMA50则下个交易日开盘退出；止损只能收紧，不放宽',
    }
    action, state = '观望', 'watching'
    risk = _flat_risk('')
    if holding:
        action = '核实止损并退出已有多头' if stop_breached else '退出已有多头' if needs_exit else '管理已有多头'
        state = 'exit_pending' if needs_exit else 'holding_long'
    elif trend['direction'] == 'down':
        action = '防守/回避'
    elif trend['entry_candidate']:
        if account['status'] in {'stale', 'future'}:
            risk = _flat_risk('账户日期与行情日期不一致')
        elif known and account['guardrails']:
            risk = _flat_risk('、'.join(account['guardrails']))
        else:
            risk = naked_k_risk.build_risk_plan(
                '小仓试错', trend['entry_reference'], stop, None,
                current_drawdown_pct=account['current_drawdown_pct'] if known else 0,
                consecutive_losses=account['consecutive_losses'] if known else 0,
                config=config.risk)
            risk['planned_account_risk_pct'] = risk['suggested_gross_pct'] * risk['risk_pct'] / 100
            if not known:
                risk['current_drawdown_pct'] = None
                risk['consecutive_losses'] = None
            risk['basis'] = '按信号收盘估算的条件预算，下一交易日须按实际开盘价重算'
            if risk['suggested_gross_pct'] > 0:
                action, state = '候选入场', 'planned_long'
    zones = trend['zones']
    support = zones.get('nearest_support')
    sources = {'daily': str(daily.attrs.get('source', 'unknown'))}
    dates = {'daily': as_of}
    closes = {'daily': trend['close']}
    if weekly is not None and not weekly.empty:
        sources['weekly'] = str(weekly.attrs.get('source', 'unknown'))
        dates['weekly'] = weekly.index[-1].strftime('%Y-%m-%d')
        closes['weekly'] = float(weekly['Close'].iloc[-1])
    review = {'status': 'prior_plan_only' if previous else 'no_previous',
              'previous_signal_date': (previous or {}).get('latest_k_dates', {}).get('daily'),
              'previous_action': (previous or {}).get('action'),
              'note': '历史计划不等于成交记录；收益只从真实成交或明确回测计算'}
    return InstrumentReport(
        name=name, ticker=ticker.upper(), action=action, signal_state=state, trend=trend,
        risk_plan=risk, account=account, management=management,
        entry_trigger=trend['entry_reference'] if trend['entry_candidate'] else None,
        stop_loss=stop, target_price=None,
        risk_per_share=(trend['entry_reference'] - stop) if stop is not None else None,
        reward_to_risk=None, support=support['midpoint'] if support else None,
        resistance=trend['resistance'], position_size=risk['position_size'],
        data_sources=sources, latest_k_dates=dates, latest_closes=closes, review=review)


def apply_portfolio_limits(reports: list[InstrumentReport], config: settings.TradingConfig | None = None) -> None:
    """Allocate candidate budgets in input order, including any known actual holdings."""
    config = config or settings.TradingConfig()
    limits = config.portfolio
    known = next((r.account for r in reports if r.account['status'] == 'known'), None)
    gross = known['total_gross_pct'] if known else 0.0
    risk = known['total_account_risk_pct'] if known else 0.0
    markets = dict(known['market_gross_pct']) if known else {}
    names = dict(known['single_name_gross_pct']) if known else {}
    for report in reports:
        plan = report.risk_plan
        requested = plan['suggested_gross_pct']
        if requested <= 0:
            continue
        market = portfolio.classify_market(report.ticker)
        risk_pct = plan['risk_pct']
        capacity = min(requested, limits.max_total_gross_pct - gross,
                       limits.max_direction_gross_pct - gross,
                       limits.max_market_gross_pct - markets.get(market, 0),
                       limits.max_single_name_gross_pct - names.get(report.ticker, 0),
                       max(0, limits.max_total_account_risk_pct - risk) / risk_pct * 100)
        allocated = math.floor(max(0, capacity) * 10 + 1e-9) / 10
        plan['suggested_gross_pct'] = allocated
        actual_risk = allocated * risk_pct / 100
        plan['planned_account_risk_pct'] = actual_risk
        plan['effective_account_risk_pct'] = actual_risk
        plan['portfolio_basis'] = 'actual_plus_candidates' if known else 'hypothetical_candidates_only'
        if allocated < requested:
            plan['guardrails'].append('组合剩余容量限制（按输入顺序分配候选预算）')
        if allocated <= 0:
            report.action, report.signal_state = '观望', 'watching'
            plan['status'] = 'blocked'
        report.position_size = plan['position_size'] = f'候选新仓上限 {allocated:.1f}%（实际开盘重算；计划风险 {actual_risk:.2f}%）'
        gross += allocated
        risk += actual_risk
        markets[market] = markets.get(market, 0) + allocated
        names[report.ticker] = names.get(report.ticker, 0) + allocated
