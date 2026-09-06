# Short-history Trend Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development; implement and review the bounded backtest task separately from trend/report integration.

**Goal:** Support 55–204-bar instruments with explicit EMA20/50 rules, then rerun DRAM/FOTO.
**Architecture:** One causal trend function, shared rule version and minimum-history constants; existing planner and risk constraints remain authoritative.
**Tech Stack:** Existing Python, pandas, numpy, unittest; no new dependencies.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-09-06-short-history-trend-design.md`.
- Below55: unknown. 55–204: Close > EMA20 > EMA50 and positive five-day EMA50 slope for up; inverse for down. At205+: existing full rule unchanged.
- Keep unknown holdings unknown, next-open fills, costs, pressure-space and risk constraints. No shorts, fabricated EMA200, future lookahead or profitability claims.
- Expose `history_mode` (`insufficient`, `short`, `full`), `daily_rows`, `direction_basis`; keep `status=ready` for both eligible modes. Shared constants: `MIN_HISTORY=55`, `FULL_HISTORY=205`, `RULE_VERSION='technical-trend-v2'` in `naked_k_trend.py`.
- Preserve existing zone-display edits and old reports/journals. Worktree creation was sandbox-denied; execute in place, without push or destructive Git operations.

## Task 1: Shared trend and report integration

Files: `naked_k_trend.py`, `naked_k_analysis.py`, `naked_k_planner.py`, their existing tests.

- [x] Add failing tests for54/55/199/200/204/205 mode boundaries, short up/down/transition, candidate and repeated breakthrough, full-rule preservation, safe account plans, version consistency and report warnings.
- [x] Run `python3 -m unittest tests.test_naked_k_trend tests.test_naked_k_analysis tests.test_naked_k_planner` and confirm expected failures.
- [x] Implement constants and selection using `fast, slow = (ema20, ema50) if short else (ema50, ema200)`; apply unchanged direction comparisons/slope and downstream guards. Return mode/count/basis metadata. Replace hardcoded version strings with shared version; short-mode warning must not block eligible candidates; below55 still blocked.
- [x] Rerun the same tests and review the diff for rule and report consistency.

## Task 2: Production backtest and comparisons

Files: `naked_k_backtest.py`, `tests/test_naked_k_backtest.py`.
Consumes: Task1 constants and unchanged `analyze_trend(frame)` / `evaluate_entry(signal, open_price)` API; additional mode metadata.

- [x] Add failing tests for default55-bar preheat/56th-bar execution, reject54-bar production preheat, allow55-bar walk-forward, causal per-prefix transition and unavailable EMA50/200 comparison.
- [x] Run `python3 -m unittest tests.test_naked_k_backtest` to confirm failures.
- [x] Import shared constants, default and validate warmup against `MIN_HISTORY`. Expose mode/basis/count in audit evidence and rule version metadata. EMA comparison at an initial signal with missing EMA200 returns `status='not_computable'`, `metrics=None`, empty equity and explicit reason. Completed comparisons retain existing numerical behavior with `status='completed'`. Walk-forward aggregation propagates unavailable comparisons rather than stitching empty equity into cash performance.
- [x] Rerun backtest tests and self-review execution dates, costs and caller handling of missing comparison metrics.

## Task 3: Documentation and acceptance

Files: `README.md`, `AGENTS.md`, `CHANGELOG.md`, approved spec and this ledger; new artifacts under `reports/`.

- [x] Document short-mode formulas, 55-bar minimum, v2 migration, explicit modes,56-bar minimum replay and unavailable long benchmark; retain historical v1 release notes.
- [x] Run `python3 -m unittest discover -v` and `git diff --check`; independently review the combined diff.
- [x] Run `python3 naked_k_analysis.py RKLB NVDA DRAM FOTO --json --report-path reports/trend_RKLB_NVDA_DRAM_FOTO_20260906_v2.md` saving stdout JSON beside it.
- [x] Verify modes, rows, dates, JSON/Markdown/journal/audit equality, and RKLB/NVDA unchanged technical decisions against prior saved run; inspect DRAM/FOTO results without assuming new support creates a buy signal.
- [x] Mark acceptance results and any economic-validation limits in this ledger; present results and report link.

## Execution evidence

- Baseline:261 offline tests passed; core regression tests first failed against v1, then35 passed with v2. Backtest implementer reported19 scoped tests and272 total passing.
- Corrected one mature-history test fixture so its close actually lies between EMA50 and EMA20; no production rule adjustment was needed.
- Live rerun: `reports/trend_RKLB_NVDA_DRAM_FOTO_20260906_v2.md` / `.json`, all daily bars through2026-09-04. DRAM108 bars: short/up/观望; FOTO69 bars: short/down/防守回避. Both lack new-entry candidates.
- RKLB/NVDA: all pre-existing trend fields, actions, signal states and risk plans equal the v1 snapshot. Mode and version metadata are additive.
- Four new journal records and audit metadata agree with JSON; Markdown exactly equals JSON report text.
- Independent whole-diff review: spec compliance PASS, quality PASS, no Critical/Important findings. Final local verification:272 offline tests passed; git diff --check clean.
- Economic effectiveness remains UNVALIDATED. Changes remain in the current checkout; no release or push performed.
