# Technical Trend Analysis Developer Notes

The user approved a transition from naked-candlestick-only analysis to daily swing trend assessment on 2026-09-06. The investment horizon is several weeks to several months. Keep the `naked_k_analysis.py` CLI name for continuity.

## Scope

- `naked_k_trend.py`: pure EMA20/50/200, Wilder ATR14/ADX14, prior-20-bar channel and relative-volume calculations; shared trend and entry rules.
- `naked_k_zones.py`: confirmed structural support/resistance and explicit-anchor HLC3-volume VWAP approximation.
- `naked_k_planner.py`: conditional plans, existing-position handling and proposed portfolio budgets.
- `naked_k_risk.py`, `naked_k_portfolio.py`, `naked_k_config.py`: validated risk and optional read-only account state.
- `naked_k_analysis.py`: CLI, market data checks, factual news appendix, Markdown/JSON/journal/audit.
- `naked_k_backtest.py`: offline long/cash replay using the exact same trend and entry functions, explicit costs, daily marked equity and comparisons.
- `westock_wrapper.py`: westock CLI -> Tencent -> Yahoo chart -> yfinance; preserve ticker normalization, adjustment metadata, UTC-internal minute data and provider limits.

## Rules

- Automatically use ponytail full for coding tasks. Prefer existing pandas/numpy and stdlib; no broad indicator frameworks or duplicated scoring engines.
- Add failing tests before changing signal or risk behavior. Tests never use the live network.
- Preserve data quality, closed-bar checks, cross-timeframe adjustment warnings and UTC timestamp regressions.
- Use only data available at the signal timestamp. Channels exclude the signal bar; structural pivots are available only after confirmation.
- Trend strength is not direction, a risk multiple is not a target prediction, and a rule threshold is not a calibrated probability.
- News contains source/date/title metadata only and must never alter technical actions; failure must preserve the technical report. Do not restore LLM direction synthesis or institutional-flow inference.
- Unknown account holdings or drawdown are not zero. Proposed budgets are not actual exposure; defensive actions never open shorts.
- Backtest uses next-open execution, adverse slippage, costs and multi-day positions. No fills at stale trigger prices, fake final liquidation or trade-count-scaled Sharpe.
- Do not claim investment effectiveness from passing unit tests or synthetic market fixtures. Real economic validation requires disclosed samples, costs and out-of-sample comparisons.
- Keep new research artifacts in `reports/`; append journal records without deleting historical schemas or reports.

## Commands

```bash
python naked_k_analysis.py 0700.HK
python naked_k_analysis.py 0700.HK TSLA --news --json
python naked_k_analysis.py 0700.HK --account-path account.json
python -m unittest discover -v
```

Design: `docs/superpowers/specs/2026-09-06-technical-trend-streamline-design.md`.
Implementation ledger: `docs/superpowers/plans/2026-09-06-technical-trend-streamline.md`.
Earlier specs describe the retired system and are retained as history, not current product requirements.
