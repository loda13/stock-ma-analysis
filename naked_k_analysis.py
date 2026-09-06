#!/usr/bin/env python3
"""Daily swing trend reports. News is a factual appendix, never a signal input."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import timedelta
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

import naked_k_audit
import naked_k_config
import naked_k_news_enhanced
import naked_k_planner
import naked_k_portfolio
import naked_k_trend
import westock_wrapper as yf

DEFAULT_JOURNAL_PATH = Path("reports/naked_k_journal.jsonl")
DEFAULT_REPORT_PATH = Path("reports/naked_k_latest.md")
DEFAULT_AUDIT_PATH = Path("reports/naked_k_audit.jsonl")
InstrumentReport = naked_k_planner.InstrumentReport
build_trade_plan = naked_k_planner.build_trade_plan
classify_market = naked_k_portfolio.classify_market



def market_timezone(market: str) -> ZoneInfo:
    """Session zone for a market, including UTC for 24/7 crypto bars."""
    return ZoneInfo(naked_k_portfolio.market_timezone_name(market))


def market_close_hour(market: str) -> int:
    return 16


def trim_to_closed_bars(
    frame: pd.DataFrame,
    market: str,
    interval: str,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    tz = market_timezone(market)
    clock = now or pd.Timestamp.now(tz=tz)
    if clock.tzinfo is None:
        clock = clock.tz_localize(tz)
    else:
        clock = clock.tz_convert(tz)

    # Reject future observations before deciding whether the current bar is closed.
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is None:
        stamp_zone = "UTC" if interval == "1h" else tz
        stamps = stamps.tz_localize(stamp_zone)
    frame = frame.loc[stamps <= clock]
    if frame.empty:
        return frame

    if market == "crypto" and interval in {"1d", "1wk", "1mo"}:
        stamps = pd.DatetimeIndex(frame.index)
        stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps.tz_convert("UTC")
        if interval == "1d":
            return frame.loc[[stamp.date() != clock.date() for stamp in stamps]]
        if interval == "1wk":
            current_week = clock.isocalendar()[:2]
            return frame.loc[[stamp.isocalendar()[:2] != current_week for stamp in stamps]]
        current_month = (clock.year, clock.month)
        return frame.loc[[(stamp.year, stamp.month) != current_month for stamp in stamps]]

    last_ts = pd.Timestamp(frame.index[-1])
    last_date = last_ts.date()
    current_date = clock.date()

    if interval == "1d" and last_date == current_date and clock.hour < market_close_hour(market):
        return frame.iloc[:-1]

    if interval == "1wk":
        current_week = clock.isocalendar()[:2]
        before_weekly_close = clock.weekday() < 4 or (clock.weekday() == 4 and clock.hour < market_close_hour(market))
        if before_weekly_close:
            return frame.loc[
                [pd.Timestamp(index).isocalendar()[:2] != current_week for index in frame.index]
            ]

    if interval == "1mo":
        current_month = (clock.year, clock.month)
        return frame.loc[
            [(pd.Timestamp(index).year, pd.Timestamp(index).month) != current_month for index in frame.index]
        ]

    if interval == "1h" and len(frame) > 1:
        latest_volume = pd.to_numeric(pd.Series([frame.iloc[-1].get("Volume")]), errors="coerce").iloc[0]
        if pd.notna(latest_volume) and float(latest_volume) <= 0:
            return frame.iloc[:-1]

    return frame


def load_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    frame = yf.download(ticker, period=period, interval=interval, progress=False)
    if frame is None or getattr(frame, "empty", True):
        raise ValueError(f"{ticker} {interval} 无可用数据")
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    frame = trim_to_closed_bars(frame, market=classify_market(ticker), interval=interval)
    if frame.empty:
        raise ValueError(f"{ticker} {interval} 只有未收盘K线")
    return naked_k_trend.validate_ohlcv(frame)



def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_journal_entry(
    rows: list[dict[str, Any]],
    ticker: str,
    current_daily_date: str | None = None,
) -> dict[str, Any] | None:
    matches = [row for row in rows if row.get("ticker") == ticker]
    if current_daily_date is not None:
        current_ts = pd.Timestamp(current_daily_date)
        matches = [
            row
            for row in matches
            if (
                (row.get("latest_k_dates") or {}).get("daily")
                and pd.Timestamp((row.get("latest_k_dates") or {}).get("daily")) < current_ts
            )
        ]
    return matches[-1] if matches else None



def build_data_audit_payload(ticker: str, interval: str, period: str, frame: pd.DataFrame) -> dict[str, Any]:
    latest = pd.Timestamp(frame.index[-1]).isoformat() if not frame.empty else None
    return {
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "rows": len(frame),
        "latest": latest,
        "source": str(frame.attrs.get("source", "unknown")),
        "adjustment": str(frame.attrs.get("adjustment", yf.ADJUSTMENT_UNKNOWN)),
    }


_TIMEFRAME_LABELS = {
    "daily": "日线",
    "weekly": "周线",
    "monthly": "月线",
}

# Only the timeframes that carry structural levels are compared. Intraday is
# excluded because its basis is *unobservable*, not merely different: the 1h window
# is ~5 days, and Tencent's minute endpoint caps at 120 bars (~30 sessions), so
# neither source can reach back past an ex-date. Over that window qfq and
# un-adjusted daily closes measured identical to 0.0000%, which is why the minute
# fetcher labels itself `unknown` rather than guessing. Since `unknown` never
# compares equal, including intraday would fire the warning on every A-share on
# every run and train the reader to ignore it. The structural timeframes below do
# span years — a daily-vs-monthly mismatch reached 8.9% on 600519 — so they are
# checked against each other.
_ADJUSTMENT_CHECKED_TIMEFRAMES = ("daily", "weekly", "monthly")


def detect_adjustment_conflict(
    frames: dict[str, pd.DataFrame | None],
) -> dict[str, Any] | None:
    """Report when the timeframes of one ticker do not share a price basis.

    Every timeframe runs `load_ohlcv` independently, so each walks the whole
    westock → Tencent → Yahoo chain on its own. Daily can land on Tencent qfq
    while weekly falls through to Yahoo's split-only OHLC; the candles then sit
    on different price axes, and any level read off one and applied to the other
    is wrong. Returns None when every present timeframe agrees.
    """
    present: dict[str, str] = {}
    sources: dict[str, str] = {}
    for timeframe in _ADJUSTMENT_CHECKED_TIMEFRAMES:
        frame = frames.get(timeframe)
        if frame is None or getattr(frame, "empty", True):
            continue
        present[timeframe] = str(frame.attrs.get("adjustment", yf.ADJUSTMENT_UNKNOWN))
        sources[timeframe] = str(frame.attrs.get("source", "unknown"))

    if len(present) < 2:
        return None

    # Compare every timeframe against the first. Sources are passed through
    # because `unknown` resolves only against an identical source: the westock-data
    # CLI is first in the fallback chain and so supplies every timeframe when
    # installed, and one provider cannot disagree with itself. Judging that by
    # source *identity alone* would be too coarse — Tencent picks its label per
    # request from whichever key answered, so it can legitimately return qfq daily
    # and split_only weekly, and that mismatch must still be reported.
    reference_timeframe, reference_basis = next(iter(present.items()))
    reference_source = sources[reference_timeframe]
    if all(
        yf.adjustments_comparable(reference_basis, basis, reference_source, sources[timeframe])
        for timeframe, basis in present.items()
    ):
        return None

    parts = [
        f"{_TIMEFRAME_LABELS.get(timeframe, timeframe)} "
        f"{yf.describe_adjustment(basis)}（{sources[timeframe]}）"
        for timeframe, basis in present.items()
    ]
    return {
        "bases": present,
        "sources": sources,
        "message": "；".join(parts) + " —— 口径不一致",
    }


def format_adjustment_warning(conflict: dict[str, Any] | None) -> str:
    """Render an adjustment conflict as a Markdown report line."""
    if not conflict:
        return ""
    return f"- ⚠️ 复权口径警告：{conflict['message']}，跨周期价位不可直接比较"



def _single_line(value: Any) -> str:
    return " ".join(str("" if value is None else value).split())



def _markdown_label(value: Any) -> str:
    text = escape(_single_line(value), quote=False)
    return (
        text.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _safe_http_url(value: Any) -> str:
    url = _single_line(value)
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")



def serialize_report(report: InstrumentReport) -> dict[str, Any]:
    return asdict(report)


def append_journal(path: Path, run_date: str, report: InstrumentReport) -> None:
    payload = {'run_date': run_date, **serialize_report(report)}
    # Append only: old schemas, same-day reruns and their original evidence survive.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = ''
    if path.exists() and path.stat().st_size:
        with path.open('rb') as existing:
            existing.seek(-1, 2)
            separator = '' if existing.read(1) == b'\n' else '\n'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(separator + encoded + '\n')


def _display(value: Any, decimals: int = 2) -> str:
    return '不可计算' if value is None else f'{value:.{decimals}f}'


def _news_appendix(report: InstrumentReport) -> list[str]:
    if not report.news:
        return []
    lines = ['', '### 新闻事实附录', '标题未经正文核验；只作事件提醒，不改变技术方向。']
    if report.news.get('status') != 'ok':
        lines.append('- 新闻不可用或窗口内无相关消息；技术评估保留。')
    for item in report.news.get('items', []):
        title = _markdown_label(item.get('title'))
        url = _safe_http_url(item.get('url'))
        label = f'[{title}]({url})' if url else title
        lines.append(f"- {_markdown_label(item.get('published_at'))} | {_markdown_label(item.get('publisher'))} | {label}")
    return lines


def format_report(run_date: str, reports: list[InstrumentReport], journal_path: Path,
                  config: naked_k_config.TradingConfig | None = None,
                  adjustment_conflicts: dict | None = None) -> str:
    lines = ['# 日线波段技术趋势报告', f'生成日期：{run_date}', f'日志：{journal_path}',
             f'周期：数周至数月；规则版本 {naked_k_trend.RULE_VERSION}；经济有效性 UNVALIDATED。', '']
    directions = {'up': '上涨', 'down': '下跌', 'transition': '过渡', 'unknown': '数据不足'}
    strengths = {'strong': '强', 'developing': '发展中', 'weak': '弱', 'unknown': '不可计算'}
    modes = {'short': '短历史模式：EMA20/50；长期趋势未确认',
             'full': '完整历史模式：EMA50/200', 'insufficient': '样本不足，暂停趋势定向和新仓计划'}
    for report in reports:
        trend, ind = report.trend, report.trend['indicators']
        warning = format_adjustment_warning((adjustment_conflicts or {}).get(report.ticker))
        lines += [f'## {_markdown_label(report.name)} {_markdown_label(report.ticker)}',
                  '- 数据源：' + '；'.join(f'{k} `{v}`' for k, v in report.data_sources.items()),
                  *([warning] if warning else []),
                  f"- 最新K线：日线 {report.latest_k_dates['daily']} 收 {_display(trend['close'])}",
                  f"- 评估依据：{modes[trend['history_mode']]}；完整日K {trend['daily_rows']} 根",
                  f"- 趋势：{directions[trend['direction']]}；ADX强度：{strengths[trend['strength']]}",
                  f'- 当前计划：{report.action}；状态：{report.signal_state}',
                  f"- EMA20 / EMA50 / EMA200：{_display(ind.get('ema20'))} / {_display(ind.get('ema50'))} / {_display(ind.get('ema200'))}",
                  f"- EMA50近5日变化：{_display(ind.get('ema50_slope'))}；ADX14：{_display(ind.get('adx'))}（强度不代表方向）",
                  f"- ATR14：{_display(ind.get('atr'))}；ATR%：{_display(ind.get('atr_pct'))}%",
                  f"- 前20日高 / 低：{_display(ind.get('channel_high'))} / {_display(ind.get('channel_low'))}；首次收盘突破：{'是' if trend['breakout'] else '否'}",
                  f"- 相对成交量：{_display(ind.get('relative_volume'))}倍（相对之前20日；只作辅助）"]
        for key, label in [('nearest_support', '支撑'), ('nearest_resistance', '压力')]:
            zone = trend['zones'].get(key)
            bounds = f"{_display(zone['lower'])}–{_display(zone['upper'])}" if zone else '未识别'
            lines.append(f'- 结构{label}区间：{bounds}')
        vwap = trend['zones'].get('anchored_vwap')
        if vwap:
            lines.append(f"- 锚定VWAP：{_display(vwap['value'])}；锚点 {vwap['anchor_date']}，确认于 {vwap['confirmed_at']}（HLC3成交量近似）")
        if report.signal_state == 'planned_long':
            lines += [f"- 候选入场：仅信号日后的下一交易日开盘有效，盘中不追补；参考收盘 {_display(report.entry_trigger)}，最高开盘价 {_display(trend['max_entry'])}",
                      f'- 初始止损：{_display(report.stop_loss)}；按实际开盘重算风险与首个压力位空间，不追跳空。',
                      f'- 仓位预算：{report.position_size}',
                      '- 退出：收盘跌破EMA50后下一交易日开盘退出；结构移动止损只能上移。']
        else:
            lines.append('- 新仓：0%；当前没有可执行的新仓计划。')
        if report.risk_plan.get('guardrails'):
            lines.append('- 风控限制：' + '；'.join(report.risk_plan['guardrails']))
        if report.account['status'] == 'known':
            lines.append(f"- 账户快照：{report.account['as_of']}；实际总仓位 {_display(report.account['total_gross_pct'])}%；实际账户风险 {_display(report.account['total_account_risk_pct'])}%")
        else:
            lines.append(f"- 账户状态：{report.account['status']}；真实持仓未知；{report.account['note']}")
        if report.management['holding_status'] == 'held':
            lines.append(f"- 已有多头：{'下一交易日开盘退出' if report.management['exit_next_open'] else '继续按趋势管理'}；建议保护位 {_display(report.management['suggested_stop'])}（账户原止损 {_display(report.management['previous_stop'])}）")
        else:
            lines.append('- 持仓处理：' + report.management['note'])
        if report.management.get('stop_breached'):
            lines.append('- 止损异常：收盘已触及或跌破账户原保护价，但快照仍有持仓；核实订单并安排退出，不能假设已成交。')
        reason_labels = {'trend_up': '价格与均线支持上涨趋势', 'trend_down': '价格与均线支持下跌趋势',
                         'trend_transition': '趋势处于过渡状态', 'trend_unknown': '样本不足，趋势不可评估',
                         'fresh_breakout': '首次收盘突破前20日高点', 'no_fresh_breakout': '没有新的首次突破',
                         'resistance_within_1r': '首个压力区距离不足1R'}
        lines += ['- 依据：' + '；'.join(reason_labels.get(x, str(x)) for x in trend['reasons']),
                  '- 目标：不预测固定目标价；R倍数只是风险尺度。',
                  '- 复盘：' + report.review['note']]
        if report.data_quality.get('warnings'):
            lines.append('- 数据限制：' + '；'.join(report.data_quality['warnings']))
        lines.extend(_news_appendix(report))
        lines.append('')
    proposal = naked_k_portfolio.evaluate_portfolio_exposure(reports, (config or naked_k_config.TradingConfig()).portfolio)
    lines += ['## 候选预算汇总',
              f"- 候选新仓合计 {_display(proposal['total_gross_pct'])}%；候选账户风险 {_display(proposal['total_account_risk_pct'])}%。",
              '- 以上为条件预算，不能当作实际成交或真实账户暴露；资金按输入顺序分配。',
              '- 指标描述当前趋势，不提供经验证的未来涨跌概率。']
    return '\n'.join(lines)



def _next_possible_open(signal_date: str, market: str) -> pd.Timestamp:
    """Conservative deadline: ordinary weekdays only, no holiday calendar assumed."""
    day = pd.Timestamp(signal_date).date() + timedelta(days=1)
    if market != 'crypto':
        while day.weekday() >= 5:
            day += timedelta(days=1)
    hour, minute = {'us': (9, 30), 'hk': (9, 30), 'cn': (9, 30),
                    'kr': (9, 0), 'crypto': (0, 0)}[market]
    return pd.Timestamp(day).tz_localize(market_timezone(market)).replace(hour=hour, minute=minute)


def run_analysis(tickers: list[tuple[str, str]], journal_path: Path,
                 config: naked_k_config.TradingConfig | None = None,
                 audit_path: Path | None = None, *, account_state: dict | None = None,
                 news: bool = False, news_lookback_days: int = 7, news_max_items: int = 12,
                 news_get=None, news_search_factory=None, now: pd.Timestamp | None = None) -> tuple[str, list[InstrumentReport]]:
    if not tickers or news_lookback_days <= 0 or news_max_items <= 0:
        raise ValueError('tickers required; news limits must be positive')
    symbols = [symbol.strip().upper() for _, symbol in tickers]
    if any(not s for s in symbols) or len(set(symbols)) != len(symbols):
        raise ValueError('tickers must be nonempty and unique')
    config = config or naked_k_config.TradingConfig()
    audit = naked_k_audit.AuditLogger(audit_path)
    rows, reports, conflicts = load_journal(journal_path), [], {}
    clock = pd.Timestamp.now(tz='Asia/Shanghai') if now is None else pd.Timestamp(now)
    if clock.tzinfo is None:
        clock = clock.tz_localize('Asia/Shanghai')
    run_date = clock.isoformat()
    audit.info('run_started', ticker_count=len(tickers), schema_version=naked_k_trend.RULE_VERSION)
    for (name, _), ticker in zip(tickers, symbols):
        try:
            daily = trim_to_closed_bars(load_ohlcv(ticker, '1d', '3y'), classify_market(ticker), '1d', now=clock)
            audit.info('data_loaded', **build_data_audit_payload(ticker, '1d', '3y', daily))
            try:
                weekly = load_ohlcv(ticker, '1wk', '5y')
                audit.info('data_loaded', **build_data_audit_payload(ticker, '1wk', '5y', weekly))
            except Exception as exc:
                weekly = None
                audit.warning('data_unavailable', ticker=ticker, interval='1wk', error_type=type(exc).__name__)
            conflicts[ticker] = detect_adjustment_conflict({'daily': daily, 'weekly': weekly})
            previous = latest_journal_entry(rows, ticker, daily.index[-1].strftime('%Y-%m-%d'))
            report = build_trade_plan(name, ticker, daily, weekly, previous, config=config, account_state=account_state)
            warnings = []
            if conflicts[ticker]:
                warnings.append('周线与日线复权口径不同；交易价位只使用日线，周线不参与信号')
                audit.warning('adjustment_basis_conflict', ticker=ticker, **conflicts[ticker])
            if daily.attrs.get('adjustment', 'unknown') == 'unknown':
                warnings.append('行情复权口径未知，实盘价格需核对')
            if report.trend['status'] != 'ready':
                warnings.append(f'不足{naked_k_trend.MIN_HISTORY}根完整日K，暂停趋势定向和新仓计划')
            elif report.trend['history_mode'] == 'short':
                warnings.append('短历史规则未经经济验证；长期趋势未确认')
            deadline = _next_possible_open(report.latest_k_dates['daily'], classify_market(ticker))
            if report.signal_state == 'planned_long' and clock >= deadline:
                warnings.append('候选开盘窗口已过或交易日历未核实，暂停沿用旧信号；需核对节假日和下一有效交易时点')
                report.action, report.signal_state = '观望', 'watching'
                report.risk_plan = naked_k_planner._flat_risk(warnings[-1])
                report.position_size = report.risk_plan['position_size']
            age = (clock.tz_convert(market_timezone(classify_market(ticker))).date() - daily.index[-1].date()).days
            if age > 7:
                warnings.append('行情超过7个自然日未更新，暂停新仓计划（保守新鲜度检查）')
                report.action, report.signal_state = '观望', 'watching'
                report.risk_plan = naked_k_planner._flat_risk(warnings[-1])
                report.position_size = report.risk_plan['position_size']
            report.data_quality = {'daily_rows': len(daily), 'age_calendar_days': age, 'candidate_open_deadline': deadline.isoformat(),
                                   'calendar_status': 'weekday_schedule_only', 'warnings': warnings,
                                   'adjustment': daily.attrs.get('adjustment', 'unknown')}
            if news:
                try:
                    collected = naked_k_news_enhanced.collect_news_enhanced(
                        name, ticker, lookback_days=news_lookback_days, fallback_days=news_lookback_days,
                        max_items=news_max_items, get=news_get, search_factory=news_search_factory)
                    report.news = {'status': collected.get('status', 'unavailable'), 'items': [
                        {key: item.get(key) for key in ('title', 'publisher', 'published_at', 'url')}
                        for item in collected.get('items', [])]}
                except Exception as exc:
                    report.news = {'status': 'unavailable', 'items': []}
                    audit.warning('news_unavailable', ticker=ticker, error_type=type(exc).__name__)
            reports.append(report)
        except Exception as exc:
            audit.error('run_failed', ticker=ticker, error_type=type(exc).__name__)
            raise
    naked_k_planner.apply_portfolio_limits(reports, config)
    for report in reports:
        append_journal(journal_path, run_date, report)
        audit.info('plan_generated', ticker=report.ticker, action=report.action,
                   signal_state=report.signal_state, account_status=report.account['status'],
                   rule_version=naked_k_trend.RULE_VERSION, history_mode=report.trend['history_mode'],
                   daily_rows=report.trend['daily_rows'], direction_basis=report.trend['direction_basis'])
    text = format_report(run_date, reports, journal_path, config, conflicts)
    audit.info('run_completed', ticker_count=len(reports), schema_version=naked_k_trend.RULE_VERSION)
    return text, reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='生成日线波段技术趋势报告（数周至数月）')
    parser.add_argument('tickers', nargs='+', metavar='TICKER')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--journal-path', default=str(DEFAULT_JOURNAL_PATH))
    parser.add_argument('--report-path', default=str(DEFAULT_REPORT_PATH))
    parser.add_argument('--audit-path', default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument('--config-path')
    parser.add_argument('--account-path', help='只读账户JSON快照；缺失时真实持仓未知')
    parser.add_argument('--news', action='store_true', help='仅附新闻日期、来源和标题，不调用模型')
    parser.add_argument('--news-lookback-days', type=int, default=7)
    parser.add_argument('--news-max-items', type=int, default=12)
    removed = {'--llm', '--llm-base-url', '--llm-model', '--news-model'}
    if any(arg.split('=')[0] in removed for arg in sys.argv[1:]):
        parser.error('LLM与新闻模型选项已移除；--news 现在只提供事实附录')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = naked_k_config.load_trading_config(args.config_path)
        state = json.loads(Path(args.account_path).read_text(encoding='utf-8')) if args.account_path else None
        text, reports = run_analysis([(t, t) for t in args.tickers], Path(args.journal_path),
                                     config=config, audit_path=Path(args.audit_path) if args.audit_path else None,
                                     account_state=state, news=args.news,
                                     news_lookback_days=args.news_lookback_days, news_max_items=args.news_max_items)
        path = Path(args.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        if args.json:
            print(json.dumps({'schema_version': naked_k_trend.RULE_VERSION, 'report': text,
                              'items': [serialize_report(r) for r in reports],
                              'report_path': str(path), 'audit_path': args.audit_path or None},
                             ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(text)
        return 0
    except (ValueError, OSError) as exc:
        print(f'无法生成报告：{exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
