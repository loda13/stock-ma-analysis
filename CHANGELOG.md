# Changelog


## v4.0.0 — 2026-09-06

- Replace naked-K/AI/Smart Money narratives with a shared daily EMA20/50/200, ADX14, ATR14 and prior-channel trend engine.
- Retain confirmed structure, explicit-anchor approximate VWAP, data audit, factual news and append-only journals.
- Separate hypothetical candidate budgets from optional, dated actual account holdings; reject invalid/stale account inputs and retired configuration.
- Remove institutional-flow inference, pseudo volume profiles, model direction synthesis and their unused code/tests.
- Replace single-bar phantom-short backtests with cost-aware long/cash multi-day replay and transparent comparisons.
- New schema: `technical-trend-v1`. Economic effectiveness remains UNVALIDATED.
- Breaking migration: removed model CLI flags, smart-money configuration and old Python advisor interfaces; retain the main CLI name and historical journal records.
- Verify 260 offline tests; model next-open fills, gap stops, costs, portfolio limits and independent walk-forward window positions explicitly.

This file keeps a concise history of user-visible changes. Git history and tags
remain the detailed implementation archive.

### Changes since v3.5.1 before the trend migration

- Require one or more explicit ticker arguments for every CLI run.
- Remove ad-hoc ticker runners and deprecated, production-unreachable trade-flow modules.
- Consolidate project documentation and keep generated reports outside the maintained tree.
- Enforce the requested news lookback after provider merging, with an explicit low-freshness fallback.
- Keep final actions, trader briefs, AI payloads, position guidance, and daily summaries semantically aligned.
- Normalize yfinance single-ticker MultiIndex data before downstream OHLCV analysis.

## v3.5.1 — 2026-08-19

- Fixed timezone-aware versus timezone-naive comparisons in recent-signal windows.
- Hardened intraday proxy collection and its public entry-point coverage.
- Removed unused intraday-flow payload fields while preserving report keys.

## v3.4.1 — 2026-08-17

- Added deterministic OHLCV volume/price proxy signals and signal-freshness handling.
- Corrected zero-position and avoid-action exposure semantics.
- These rules are uncalibrated proxy evidence; they do not identify institutions or provide probabilities.

## v3.3.1 — 2026-08-04

- Extended SEC material-event collection to both 8-K and 6-K filings.
- Preserved distinct same-day filings by accession-aware titles.
- Kept provider failures nonfatal to the technical report.

## v3.2.0 — 2026-07-21

- Added optional Finnhub news and company alias/provider metadata.
- Added multi-source news relevance scoring and provider-specific fallbacks.

## v3.1.0

- Added Korean market normalization and session timezone handling.
- Hardened the two-pass news evidence boundary and deterministic fallback.

## v3.0.0

- Split the professional naked-K workflow into focused planning, structure,
  trade, risk, portfolio, audit, backtest, and optional LLM/news modules.

## v2.0.0

- Removed the former multi-indicator advisor and focused the repository on
  candlesticks, OHLCV price structure, triggers, invalidation, targets, and journaling.
