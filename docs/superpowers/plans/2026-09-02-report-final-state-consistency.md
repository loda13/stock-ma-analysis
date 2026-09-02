# Report Final-State Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every report surface describes the final post-news, post-guardrail action without ambiguous defensive wording.

**Architecture:** Keep technical calculation and news synthesis unchanged. Rebuild the existing derived `trader_brief` and `ai_assistant` payloads at the two existing final-action boundaries, and make the current formatter choose wording from the plan's actual price direction.

**Tech Stack:** Python standard library, `unittest`, existing naked-K modules only.

## Global Constraints

- Preserve the current uncommitted fixes and do not commit or push.
- Keep deterministic OHLCV/risk fields authoritative.
- Do not add indicators, dependencies, or live-network tests.

---

### Task 1: Refresh derived report fields after final-action changes

**Files:**
- Modify: `naked_k_planner.py`
- Modify: `naked_k_analysis.py`
- Test: `tests/test_naked_k_analysis.py`

**Interfaces:**
- Consumes: an `InstrumentReport` after news synthesis or portfolio guardrails.
- Produces: `refresh_report_derivatives(report)` with synchronized `trader_brief` and `ai_assistant`, preserving an existing `llm_commentary` result.

- [x] **Step 1: Write the failing integration assertions**

```python
self.assertEqual(report.trader_brief["交易计划"].split("；", 1)[0], f"当前机会：{report.action}")
self.assertEqual(report.ai_assistant["engine_plan"]["action"], report.action)
self.assertEqual(journal_row["trader_brief"], report.trader_brief)
self.assertEqual(journal_row["ai_assistant"], report.ai_assistant)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_news_pipeline_snapshots_then_deliberates_and_keeps_legacy_llm_separate -v`

Expected: FAIL because the fixture's pre-news derived fields do not describe the synthesized action.

- [x] **Step 3: Add the smallest shared refresh helper and call it at both mutation boundaries**

```python
def refresh_report_derivatives(report: InstrumentReport) -> None:
    commentary = report.ai_assistant.get("llm_commentary")
    report.trader_brief = naked_k_interpreter.build_trader_brief(report)
    report.ai_assistant = naked_k_ai.build_ai_trading_assistant(report)
    if commentary is not None:
        report.ai_assistant["llm_commentary"] = commentary
```

Call it after per-instrument news synthesis and after portfolio guardrails, before journal serialization.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_news_pipeline_snapshots_then_deliberates_and_keeps_legacy_llm_separate -v`

Expected: PASS.

### Task 2: Make trader-brief paths directional and meaningful without a target

**Files:**
- Modify: `naked_k_interpreter.py`
- Test: `tests/test_naked_k_interpreter.py`

**Interfaces:**
- Consumes: `entry_trigger`, `stop_loss`, and optional `target_price`.
- Produces: exact `跌破`/`突破` invalidation wording and a confirmation-only path when no first target exists.

- [x] **Step 1: Write failing long, defensive, and no-target assertions**

```python
self.assertIn("跌破失效位 99.0", long_brief["可能交易路径"][1])
self.assertIn("突破失效位 110.0", defensive_brief["可能交易路径"][1])
self.assertNotIn("暂无第一目标", defensive_brief["可能交易路径"][0])
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_naked_k_interpreter -v`

Expected: FAIL on generic `跌破/突破` and `暂无第一目标` wording.

- [x] **Step 3: Select wording from the actual stop/trigger relationship**

```python
invalidation_verb = "跌破" if stop_loss < entry_trigger else "突破" if stop_loss > entry_trigger else "触及"
path_a = (
    f"路径A：价格触发 {entry_trigger} 后延续，先看 {target}"
    if target is not None
    else f"路径A：价格触发 {entry_trigger} 后，等待收盘确认再评估"
)
```

- [x] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest tests.test_naked_k_interpreter -v`

Expected: PASS.

### Task 3: Separate reduce and avoid actions in the daily summary

**Files:**
- Modify: `naked_k_analysis.py`
- Test: `tests/test_naked_k_analysis.py`

**Interfaces:**
- Consumes: final `report.action` values.
- Produces: independent `需要减仓` and `需要回避` summary lines.

- [x] **Step 1: Write the failing summary assertion**

```python
self.assertIn("需要减仓：公司-REDUCE", today)
self.assertIn("需要回避：公司-AVOID", today)
self.assertNotIn("需要回避：公司-REDUCE", today)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_format_report_separates_reduce_and_avoid_actions -v`

Expected: FAIL because both actions currently share the avoid line.

- [x] **Step 3: Render one line per action**

```python
f"- 需要减仓：{', '.join(item.name for item in ranked if item.action == '减仓') or '无'}",
f"- 需要回避：{', '.join(item.name for item in ranked if item.action == '回避') or '无'}",
```

- [x] **Step 4: Run focused and full verification**

Run: `python -m unittest tests.test_naked_k_analysis tests.test_naked_k_interpreter -v`

Run: `python -m unittest discover -v`

Run: `git diff --check`

Expected: all tests pass and no whitespace errors.

### Task 4: Clarify that reduce guidance applies only to existing longs

**Files:**
- Modify: `naked_k_trade.py`
- Modify: `naked_k_synthesis.py`
- Test: `tests/test_naked_k_analysis.py`
- Test: `tests/test_naked_k_synthesis.py`

**Interfaces:**
- Consumes: a final `减仓` action.
- Produces: position guidance that cannot be mistaken for permission to open a new 10% position.

- [x] **Step 1: Write failing assertions for base and clamped guidance**

```python
self.assertIn("仅处理已有多头，不新建仓", guidance)
self.assertIn("仅处理已有多头，不新建仓", report.position_size)
```

- [x] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_reduce_position_guidance_is_not_a_new_position tests.test_naked_k_synthesis.NakedKSynthesisTests.test_corroborated_reduce_clamps_residual_gross_and_account_risk_to_baseline -v`

Expected: FAIL because both current strings omit the existing-holdings qualifier.

- [x] **Step 3: Add the same concise qualifier to both reduce guidance paths**

```python
return "降至10%以内（仅处理已有多头，不新建仓）"
```

Apply the qualifier to the dynamically clamped residual string as well.

- [x] **Step 4: Run focused and full verification**

Run: `python -m unittest tests.test_naked_k_analysis tests.test_naked_k_synthesis -v`

Run: `python -m unittest discover -v`

Expected: all tests pass.

### Task 5: Normalize yfinance single-ticker MultiIndex columns

**Files:**
- Modify: `westock_wrapper.py`
- Test: `tests/test_westock_wrapper.py`

**Interfaces:**
- Consumes: the current yfinance single-ticker frame whose columns are `(Price, Ticker)` pairs.
- Produces: the wrapper's existing flat OHLCV-compatible column contract.

- [x] **Step 1: Write a failing offline reproduction**

```python
columns = pd.MultiIndex.from_tuples([
    ("Open", "1810.HK"), ("High", "1810.HK"), ("Low", "1810.HK"),
    ("Close", "1810.HK"), ("Volume", "1810.HK"),
])
self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_westock_wrapper.WestockWrapperTests.test_fetch_yfinance_flattens_single_ticker_multiindex_columns -v`

Expected: FAIL because `fetch_yfinance()` currently returns the MultiIndex unchanged.

- [x] **Step 3: Flatten the level that owns the OHLCV labels**

```python
if isinstance(frame.columns, pd.MultiIndex):
    for level in range(frame.columns.nlevels):
        labels = frame.columns.get_level_values(level)
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(labels):
            frame.columns = labels
            break
```

- [x] **Step 4: Run focused and full verification, then repeat the live run**

Run: `python -m unittest tests.test_westock_wrapper -v`

Run: `python -m unittest discover -v`

Expected: all tests pass; the live three-ticker run reaches `run_completed`.

### Task 6: Keep dual-evidence integration tests offline

**Files:**
- Modify: `tests/test_dual_evidence_integration.py`

**Interfaces:**
- Consumes: the existing `collect_intraday_flow()` path inside HK plan construction.
- Produces: deterministic `UNAVAILABLE` test behavior without contacting yfinance.

- [x] **Step 1: Confirm the current full suite emits live-network failures**

Run: `python -m unittest discover -v`

Expected before the test fix: tests pass but print yfinance proxy/connection errors from the two HK integration tests.

- [x] **Step 2: Patch the minute-bar fetcher for this test class**

```python
def setUp(self):
    patcher = patch("naked_k_intraday_flow.fetch_intraday_bars", return_value=None)
    patcher.start()
    self.addCleanup(patcher.stop)
```

- [x] **Step 3: Run the integration tests and full suite**

Run: `python -m unittest tests.test_dual_evidence_integration -v`

Run: `python -m unittest discover`

Expected: all tests pass with no live-network error text.
