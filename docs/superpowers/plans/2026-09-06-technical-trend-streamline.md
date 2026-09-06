# Technical Trend Streamline Implementation Plan

> Execute task-by-task using superpowers:subagent-driven-development, tests first. User approved the design with `go`; no further design approval is needed.

**Goal:** Replace the sprawling narrative engine with a reproducible daily swing trend report and matching long/cash backtest.

**Architecture:** A pure `naked_k_trend` module computes causal indicators and signals. The planner adds risk/account context; CLI owns data, factual news and artifacts. Backtest consumes the same pure signal, models positions and explicit costs.

**Tech Stack:** Existing Python, pandas, numpy, unittest; no new dependency.

## Global constraints

- Preserve market data fallback, adjustment metadata and closed-bar checks.
- No secrets or model requests; factual news is optional and cannot alter signals.
- No historical report deletion or rewriting. New artifacts use schema `technical-trend-v1`.
- No future observations, phantom shorts, guaranteed targets or uncalibrated win probabilities.
- Tests are offline. All generated research output belongs under reports/.
- EMA20/50/200, Wilder ATR14/ADX14, prior-20-bar channels and volume baseline; 205-bar readiness.
- User-approved changes override the old indicator-free repository scope.

## Task 1: Pure indicators and structural levels

Files: new `naked_k_trend.py`, replace `naked_k_zones.py`, `tests/test_naked_k_trend.py`, `tests/test_naked_k_zones.py`.

Interfaces: `indicator_frame(frame) -> DataFrame`; `analyze_trend(frame) -> dict`; `detect_price_zones(frame, close=None, swing_window=2) -> dict`.

- [x] Test numeric initialization, flat/zero-volume/short-history/invalid-price inputs and no-lookahead prefixes; run `python -m unittest tests.test_naked_k_trend tests.test_naked_k_zones -v` and observe RED.
- [x] Implement SMA-seeded EMA and Wilder smoothing, prior channels, direction/strength, first breakout, max entry and trend exit. Structural pivots require right-hand confirmation; AVWAP exposes anchor and confirmation dates. Delete pseudo volume-profile calculations.
- [x] Re-run focused tests, review behavior and diff; fix findings before integration.

## Task 2: Planner, account validation and report migration

Files: `naked_k_planner.py`, `naked_k_config.py`, `naked_k_risk.py`, `naked_k_portfolio.py`, `naked_k_analysis.py`; replacement tests for these paths.

Interfaces: `build_trade_plan(name, ticker, daily, weekly=None, previous=None, *, config=None, account_state=None) -> InstrumentReport`; `run_analysis(...) -> (markdown, reports)`; `evaluate_entry(signal, open_price) -> dict` lives in pure engine and is shared with backtest.

- [x] Write offline report/account/CLI tests before changing production code. Preserve existing data-loading, timezone and adjustment tests independently of removed report fields.
- [x] Reduce config to risk/portfolio; validate finite numbers and ranges, reject removed smart-money configuration with migration guidance. Validate optional account JSON including date, holdings, drawdown and loss count; unknown is not zero.
- [x] Plan/report use one trend snapshot and explicit candidate/held/unknown states. Portfolio output separates hypothetical new positions from supplied holdings; known over-limit account blocks new exposure.
- [x] Keep existing CLI/ticker/data/audit functions; replace narrative formatter and news synthesis with a factual appendix. Remove old model CLI arguments with explicit error guidance. Append new journal records without rewriting historical entries.
- [x] Run focused CLI/config/risk/portfolio/planner tests and verify JSON/Markdown/journal alignment.

## Task 3: Executable long/cash validation

Files: `naked_k_backtest.py`, `tests/test_naked_k_backtest.py`.

- [x] Regression tests for gap entry rejection, gap stop execution, defensive no-short, multiple-day hold, no duplicate positions and cost deductions; observe RED.
- [x] Implement daily position state machine using shared `analyze_trend`/`evaluate_entry`. Use next open for entry/EMA exit, prior-known stop for intraday stop, and only tighten stop after close. Retain open positions marked to market at sample end.
- [x] Require explicit commission/slippage; report net daily equity returns, drawdown, annualized daily Sharpe (zero risk-free assumption), exposure, R outcomes and independent sample dates.
- [x] Add same-period buy/hold and EMA50/200 long/cash comparisons and frozen chronological non-overlapping walk-forward windows; unavailable external benchmark remains null. Remove old misleading Monte Carlo output.
- [x] Run focused tests and review state ordering and shared-signal consistency.

## Task 4: Delete dead paths, document and verify

Files: remove unused old AI/smart-money/synthesis/context/setup modules and their exclusive tests; update README, AGENTS, config.example.json, account.example.json, CHANGELOG.

- [x] Search every remaining import before deleting modules. Retain news collection/provider tests and independent market-data audit coverage.
- [x] Run `python -m unittest discover -v`, `git diff --check`, CLI help and offline end-to-end report/backtest artifact generation.
- [x] Review the whole change for schema compatibility, actual deletion, investment semantic honesty and reproducibility; fix critical findings and rerun affected checks.
- [x] Record validation evidence in reports/ and this ledger. Do not claim economic effectiveness from synthetic fixtures.

## Completion evidence — 2026-09-06

- User-approved design implemented on `codex/technical-trend`; no push or deployment.
- Final offline suite: `python -m unittest discover -v` — **260 tests, PASS**.
- `git diff --check`, compilation, both CLI help commands — PASS.
- Offline report/JSON/journal/audit, explicit-cost backtest CLI and two frozen walk-forward windows — PASS.
- Independent final review: PASS, no remaining Critical/Important findings. Fixed pressure-zone lower edge, missing volume, expired candidates, breached protective stops, ticker whitespace, premature risk rounding and omitted window-end positions.
- Existing provider fallback/timezone/adjustment and factual-news tests retained; retired module tests removed with those modules.
- Evidence and synthetic samples: `reports/technical_trend_validation_20260906/`. Real market economic effectiveness remains **UNVALIDATED**.
