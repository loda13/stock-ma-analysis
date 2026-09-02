# Report Semantic Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the enhanced news window, structured zero-exposure state, and multi-ticker summary semantically consistent.

**Architecture:** Fix each value at its shared producer instead of patching the generated Markdown. The enhanced collector selects fresh items after merging providers, position guidance mirrors the authoritative risk plan, and the final summary joins every matching ticker.

**Tech Stack:** Python, pandas, stdlib `unittest`

## Global Constraints

- Preserve the positional ticker CLI and indicator-free naked-K scope.
- Tests must not access the live network.
- Do not add dependencies or unrelated abstractions.
- Do not invent targets or R/R for `watching`, `flat`, or defensive-only states.
- Do not commit or push; the user requested a local fix only.

---

### Task 1: Enforce the requested news window after provider merge

**Files:**
- Modify: `naked_k_news_enhanced.py:250-292`
- Test: `tests/test_naked_k_news_enhanced.py`

**Interfaces:**
- Consumes: `collect_news_enhanced(..., now, lookback_days, fallback_days, max_items)` and provider-normalized `published_at` values.
- Produces: a collection containing only main-window items when any exist; otherwise fallback-window items labeled `low_freshness`.

- [x] **Step 1: Write failing public-API tests**

```python
def test_merge_drops_old_provider_items_when_fresh_items_exist(self):
    # AkShare returns one fresh and one old but relevant Xiaomi item.
    # Assert that only the fresh item reaches result["items"].

def test_merge_labels_stale_only_items_as_low_freshness(self):
    # AkShare returns one item outside lookback_days but inside fallback_days.
    # Assert window_days == fallback_days and every item is low_freshness.
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_naked_k_news_enhanced.EnhancedNewsCollectionTests.test_merge_drops_old_provider_items_when_fresh_items_exist tests.test_naked_k_news_enhanced.EnhancedNewsCollectionTests.test_merge_labels_stale_only_items_as_low_freshness -v`

Expected: the first test includes the old item and the second incorrectly reports `fresh`.

- [x] **Step 3: Implement the minimum merged-window selection**

```python
fresh_cutoff = as_of - pd.Timedelta(days=lookback_days)
fallback_cutoff = as_of - pd.Timedelta(days=fallback_days)
fresh = [item for item in scored_candidates if fresh_cutoff <= item["published_at"] <= as_of]
fallback = [item for item in scored_candidates if fallback_cutoff <= item["published_at"] < fresh_cutoff]
if fresh:
    selected_pool, collection_freshness, window_days = fresh, "fresh", lookback_days
elif fallback:
    selected_pool, collection_freshness, window_days = fallback, "low_freshness", fallback_days
else:
    selected_pool, collection_freshness, window_days = [], "insufficient", lookback_days
selected = _deduplicate_ranked(selected_pool, max_items)
```

Use `collection_freshness` for item and collection metadata, and `window_days` in the result.

- [x] **Step 4: Verify GREEN**

Run the two focused tests from Step 2, then `python -m unittest tests.test_naked_k_news_enhanced -v`.

Expected: all pass without network access.

### Task 2: Make observation sizing agree with zero exposure

**Files:**
- Modify: `naked_k_trade.py:110-126`
- Modify: `naked_k_synthesis.py:520-529`
- Test: `tests/test_naked_k_analysis.py:2260-2270`
- Test: `tests/test_naked_k_synthesis.py:340-350`

**Interfaces:**
- Consumes: valid action plus the risk plan produced by `build_risk_plan`.
- Produces: `position_size == "0%（无新仓计划）"` whenever the final action is `观望` and risk status is `flat`.

- [x] **Step 1: Change existing assertions to the required zero-exposure wording**

```python
self.assertEqual(report.position_size, "0%（无新仓计划）")
self.assertEqual(report.position_size, report.risk_plan["position_size"])
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_bullish_trade_plan_downgrades_when_first_target_has_poor_reward_to_risk tests.test_naked_k_synthesis.NakedKSynthesisTests.test_observation_rebuilds_bullish_boundaries_and_clears_directionality -v`

Expected: both report the old `0%-10%` value.

- [x] **Step 3: Implement the shared producer fix**

```python
# naked_k_trade.build_position_guidance
else:
    return "0%（无新仓计划）"

# naked_k_synthesis._build_candidate, after setting risk_plan["position_size"]
position_size = str(risk_plan["position_size"])
```

Remove the later `position_size = "0%-10%"` override for `观望`.

- [x] **Step 4: Verify GREEN**

Run the two focused tests from Step 2, then `python -m unittest tests.test_naked_k_analysis tests.test_naked_k_synthesis -v`.

Expected: all pass and no `观望` producer emits `0%-10%`.

### Task 3: Include every observed ticker in the daily summary

**Files:**
- Modify: `naked_k_analysis.py:870-890`
- Test: `tests/test_naked_k_analysis.py`

**Interfaces:**
- Consumes: ranked `InstrumentReport` objects.
- Produces: `继续观察` containing every report whose final action is `观望`, in ranked input order.

- [x] **Step 1: Write the failing aggregation test**

```python
def test_format_report_lists_all_observation_candidates(self):
    reports = [self._integration_report("AAA"), self._integration_report("BBB")]
    text = naked_k_analysis.format_report(
        "2026-09-01 16:00:00 CST", reports, naked_k_analysis.DEFAULT_JOURNAL_PATH
    )
    today = text.split("## 今日结论", 1)[1]
    self.assertIn("继续观察：公司-AAA, 公司-BBB", today)
```

- [x] **Step 2: Verify RED**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_format_report_lists_all_observation_candidates -v`

Expected: only `公司-AAA` is displayed.

- [x] **Step 3: Replace first-match aggregation with the existing join pattern**

```python
observed_names = ", ".join(item.name for item in ranked if item.action == "观望") or "无"
```

Render `observed_names` in the `继续观察` line.

- [x] **Step 4: Verify GREEN and the full repository**

Run: `python -m unittest tests.test_naked_k_analysis.NakedKAnalysisTests.test_format_report_lists_all_observation_candidates -v`

Then run: `python -m unittest discover -v`

Expected: focused test passes and the full suite reports zero failures.

### Task 4: Re-run the three-ticker semantic acceptance check

**Files:**
- Create at runtime: `reports/naked_k_0700_1810_9992_20260901_fix_validation.md`
- Create at runtime: `reports/naked_k_journal_0700_1810_9992_20260901_fix_validation.jsonl`
- Create at runtime: `reports/naked_k_audit_0700_1810_9992_20260901_fix_validation.jsonl`

**Interfaces:**
- Consumes: `0700.HK 1810.HK 9992.HK` with the normal 14-day news workflow.
- Produces: one completed run whose structured and Markdown fields agree.

- [x] **Step 1: Run a fresh batch**

Run: `python naked_k_analysis.py 0700.HK 1810.HK 9992.HK --news --news-lookback-days 14 --news-max-items 12 --json --report-path reports/naked_k_0700_1810_9992_20260901_fix_validation.md --journal-path reports/naked_k_journal_0700_1810_9992_20260901_fix_validation.jsonl --audit-path reports/naked_k_audit_0700_1810_9992_20260901_fix_validation.jsonl`

- [x] **Step 2: Verify the artifacts**

Check one run ID, `run_started`, `run_completed`, no `run_failed`, three journal rows, zero executable gross/account risk for flat plans, no `仓位建议：0%-10%`, all observation names in `今日结论`, and no main-window collection item older than `as_of - 14 days`.

- [x] **Step 3: Report remaining limits honestly**

Keep target/R/R empty for non-actionable plans and distinguish current-run zero new exposure from any pre-existing user holdings.
