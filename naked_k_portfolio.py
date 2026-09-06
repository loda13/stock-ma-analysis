from __future__ import annotations

import math
from typing import Any

import naked_k_config


def classify_market(ticker: str) -> str:
    symbol = ticker.upper()
    if symbol.endswith(".HK"):
        return "hk"
    if symbol.endswith((".SS", ".SZ", ".BJ")):
        return "cn"
    if symbol.endswith((".KS", ".KQ")):
        return "kr"
    if symbol in {"BTC-USD", "ETH-USD", "SOL-USD"}:
        return "crypto"
    return "us"


# Single market -> IANA zone map for the whole repo. Lives here beside
# classify_market because this module imports only naked_k_config, so both the
# data layer and the CLI can reach it without a cycle. Crypto uses UTC because
# its 24/7 bars have no exchange-local session.
_MARKET_TIMEZONES = {
    "cn": "Asia/Shanghai",
    "crypto": "UTC",
    "hk": "Asia/Hong_Kong",
    "kr": "Asia/Seoul",
    "us": "America/New_York",
}
_DEFAULT_TIMEZONE = "Asia/Shanghai"


def market_timezone_name(market: str) -> str:
    """IANA zone name for a market, defaulting to the mainland session."""
    return _MARKET_TIMEZONES.get(market, _DEFAULT_TIMEZONE)


def _value(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _risk_plan(report: Any) -> dict[str, Any]:
    value = _value(report, "risk_plan", {})
    return value if isinstance(value, dict) else {}


def _direction_bucket(direction: str) -> str:
    # Defensive plans represent residual long exposure being reduced or avoided;
    # they are not naked shorts and therefore still consume the long-side cap.
    return "long" if direction == "bearish_defensive" else direction


def evaluate_portfolio_exposure(
    reports: list[Any],
    config: naked_k_config.PortfolioConfig | None = None,
) -> dict[str, Any]:
    limits = config or naked_k_config.PortfolioConfig()
    positions: list[dict[str, Any]] = []
    for report in reports:
        risk_plan = _risk_plan(report)
        gross_pct = float(risk_plan.get("suggested_gross_pct", 0.0) or 0.0)
        account_risk_pct = float(risk_plan.get("effective_account_risk_pct", 0.0) or 0.0)
        # 仅汇总非零仓位；调用方决定这些是账户快照还是候选预算。
        if gross_pct <= 0:
            continue
        ticker = str(_value(report, "ticker", ""))
        direction = _direction_bucket(str(risk_plan.get("direction", "none")))
        positions.append(
            {
                "ticker": ticker,
                "action": str(_value(report, "action", "")),
                "direction": direction,
                "market": classify_market(ticker),
                "gross_pct": gross_pct,
                "account_risk_pct": account_risk_pct,
            }
        )

    total_gross = math.fsum(position["gross_pct"] for position in positions)
    total_account_risk = math.fsum(position["account_risk_pct"] for position in positions)
    direction_gross: dict[str, float] = {}
    market_gross: dict[str, float] = {}
    single_name_gross: dict[str, float] = {}
    for position in positions:
        direction = position["direction"]
        market = position["market"]
        ticker = position["ticker"]
        direction_gross[direction] = direction_gross.get(direction, 0.0) + position["gross_pct"]
        market_gross[market] = market_gross.get(market, 0.0) + position["gross_pct"]
        single_name_gross[ticker] = single_name_gross.get(ticker, 0.0) + position["gross_pct"]

    guardrails: list[str] = []
    if total_gross > limits.max_total_gross_pct:
        guardrails.append("总仓位暴露超限")
    if total_account_risk > limits.max_total_account_risk_pct:
        guardrails.append("账户风险暴露超限")
    for direction, gross_pct in direction_gross.items():
        if direction == "long" and gross_pct > limits.max_direction_gross_pct:
            guardrails.append("多头方向暴露超限")
        elif direction == "short" and gross_pct > limits.max_direction_gross_pct:
            guardrails.append("空头方向暴露超限")
    for market, gross_pct in market_gross.items():
        if gross_pct > limits.max_market_gross_pct:
            guardrails.append(f"{market}市场暴露超限")
    for ticker, gross_pct in single_name_gross.items():
        if gross_pct > limits.max_single_name_gross_pct:
            guardrails.append(f"{ticker}单标的暴露超限")

    return {
        "status": "over_limit" if guardrails else "within_limits",
        "total_gross_pct": total_gross,
        "total_account_risk_pct": total_account_risk,
        "direction_gross_pct": direction_gross,
        "market_gross_pct": market_gross,
        "single_name_gross_pct": single_name_gross,
        "positions": positions,
        "limits": {
            "max_total_gross_pct": limits.max_total_gross_pct,
            "max_direction_gross_pct": limits.max_direction_gross_pct,
            "max_market_gross_pct": limits.max_market_gross_pct,
            "max_single_name_gross_pct": limits.max_single_name_gross_pct,
            "max_total_account_risk_pct": limits.max_total_account_risk_pct,
        },
        "guardrails": guardrails,
    }


def format_portfolio_exposure(exposure: dict[str, Any]) -> str:
    if not exposure:
        return "暂无"
    status = "超限" if exposure.get("status") == "over_limit" else "正常"
    guardrails = exposure.get("guardrails") or []
    guardrail_text = "、".join(str(item) for item in guardrails) if guardrails else "无"
    return (
        f"{status}；总仓位 {exposure.get('total_gross_pct', 0)}%；"
        f"账户风险 {exposure.get('total_account_risk_pct', 0)}%；"
        f"保护：{guardrail_text}"
    )


def assess_account(state: dict[str, Any] | None, as_of: str,
                   config: naked_k_config.TradingConfig | None = None) -> dict[str, Any]:
    """Validate a supplied long-only account snapshot; absence never means cash."""
    from datetime import date

    if state is None:
        return {'status': 'unknown', 'total_gross_pct': None, 'total_account_risk_pct': None,
                'positions': None, 'guardrails': [], 'note': '未提供账户快照；真实持仓与账户风控不可评估'}
    required = {'as_of', 'positions', 'current_drawdown_pct', 'consecutive_losses'}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError('account requires as_of, positions, current_drawdown_pct, consecutive_losses')
    try:
        snapshot_date = date.fromisoformat(state['as_of'])
        evaluation_date = date.fromisoformat(as_of[:10])
    except (TypeError, ValueError):
        raise ValueError('account as_of must be YYYY-MM-DD') from None
    drawdown = naked_k_config.number(state['current_drawdown_pct'], 'current_drawdown_pct')
    losses = state['consecutive_losses']
    if type(losses) is not int or losses < 0:
        raise ValueError('consecutive_losses must be a nonnegative integer')
    if not isinstance(state['positions'], list):
        raise ValueError('positions must be an array')
    positions, reports, seen = [], [], set()
    for position in state['positions']:
        if not isinstance(position, dict) or set(position) - {'ticker', 'gross_pct', 'account_risk_pct', 'stop_loss'}:
            raise ValueError('invalid account position fields')
        ticker = position.get('ticker')
        if not isinstance(ticker, str) or not ticker.strip() or ticker.strip().upper() in seen:
            raise ValueError('account positions require unique nonempty tickers')
        ticker = ticker.strip().upper()
        seen.add(ticker)
        gross = naked_k_config.number(position.get('gross_pct'), 'position.gross_pct')
        risk = naked_k_config.number(position.get('account_risk_pct'), 'position.account_risk_pct')
        if risk > gross:
            raise ValueError('position account risk cannot exceed long-only gross exposure')
        stop = position.get('stop_loss')
        if stop is not None:
            naked_k_config.number(stop, 'position.stop_loss', 1e-12, float('inf'))
        positions.append({**position, 'ticker': ticker, 'gross_pct': gross,
                          'account_risk_pct': risk, 'stop_loss': stop})
        reports.append({'ticker': ticker, 'action': '实际持仓', 'risk_plan': {
            'direction': 'long', 'suggested_gross_pct': gross, 'effective_account_risk_pct': risk}})
    settings = config or naked_k_config.TradingConfig()
    exposure = evaluate_portfolio_exposure(reports, settings.portfolio)
    guards = list(exposure['guardrails'])
    if drawdown >= settings.risk.max_drawdown_pct:
        guards.append('最大回撤保护')
    status = 'known' if snapshot_date == evaluation_date else ('stale' if snapshot_date < evaluation_date else 'future')
    return {**exposure, 'status': status, 'as_of': state['as_of'], 'positions': positions,
            'current_drawdown_pct': drawdown, 'consecutive_losses': losses, 'guardrails': guards,
            'note': '用户提供的账户快照，未连接券商核验' if status == 'known' else '账户日期与行情日期不一致，不用于执行建议'}
