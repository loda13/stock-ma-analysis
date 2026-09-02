"""Enhanced news collection with multi-query strategy and company name mapping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from naked_k_news import (
    _as_timestamp,
    _canonical_url,
    _dedupe_title,
    _validate_windows,
    collect_news,
)
from naked_k_news_akshare import collect_akshare_news
from naked_k_news_finnhub import collect_finnhub_news
from naked_k_news_sec import collect_sec_filings
from naked_k_news_sina import collect_sina_rolling_news


# A keyword hit in the title is worth more than one in the body. The relevance
# gate reads these back, so they must stay in sync with the scores awarded in
# _calculate_relevance_score.
_TITLE_MATCH_SCORE = 3.0
_BODY_MATCH_SCORE = 1.0


@dataclass(frozen=True)
class _SourcePolicy:
    """Per-provider ranking weight and relevance-gate rule.

    Keeping both in one record means adding a provider is a single entry here
    rather than parallel edits in a weight table and a gate branch.
    """

    quality_weight: float
    # Set for providers that emit market-wide digests: their bodies list every
    # ticker code and issuer name incidentally, so a body hit proves nothing.
    requires_title_match: bool = False
    # Set for providers already scoped to the symbol upstream.
    bypasses_gate: bool = False


_SOURCE_POLICIES = {
    "finnhub": _SourcePolicy(quality_weight=3.0, bypasses_gate=True),
    "sec_edgar": _SourcePolicy(quality_weight=5.0, bypasses_gate=True),
    "akshare_em": _SourcePolicy(quality_weight=2.0, requires_title_match=True),
    # Sina is a market-wide newswire digest already attributed by headline
    # upstream, so it arrives pre-scoped to the symbol.
    "sina": _SourcePolicy(quality_weight=2.0, bypasses_gate=True),
    "google_news_rss": _SourcePolicy(quality_weight=1.0),
    "yahoo_finance": _SourcePolicy(quality_weight=0.5),
}
_DEFAULT_SOURCE_POLICY = _SourcePolicy(quality_weight=1.0)


def load_company_names() -> dict[str, dict[str, list[str]]]:
    """Load company name mappings from JSON file."""
    config_path = Path(__file__).parent / "company_names.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _sina_aliases(
    ticker: str, company_names: dict[str, dict[str, list[str]]]
) -> list[str]:
    """Return the issuer-name aliases safe to match inside a newswire digest.

    Only the ``zh`` and ``en`` name lists are used. The ``keywords`` list holds
    deliberately loose product and person tokens ("Mi", "QQ", "盲盒") that earn
    their keep when scoring a feed already scoped to one symbol, but would
    misattribute unrelated items in a market-wide digest.
    """
    mapping = company_names.get(ticker, {})
    return [*mapping.get("zh", []), *mapping.get("en", [])]


def collect_news_enhanced(
    name: str,
    ticker: str,
    *,
    now=None,
    lookback_days: int = 7,
    fallback_days: int = 30,
    max_items: int = 12,
    search_factory=None,
    get=None,
    use_finnhub: bool = True,
    use_akshare: bool = True,
    akshare_fetch=None,
    use_sina: bool = True,
    sina_get=None,
    use_sec: bool = True,
    sec_get=None,
) -> dict[str, Any]:
    """
    Enhanced news collection with multi-query strategy and Finnhub integration.

    Performs multiple searches using company name variations and merges results.
    Includes Finnhub professional financial news when API key is available, and
    AkShare Chinese-language coverage plus Sina's minute-level newswire when the
    optional dependency is installed. For tickers with a CIK, includes SEC EDGAR
    material event filings (8-K and 6-K). Prioritizes high-quality sources and
    filters by relevance.
    """
    _validate_windows(lookback_days, fallback_days, max_items)
    as_of = _as_timestamp(now)
    company_names = load_company_names()
    queries = _generate_queries(name, ticker, company_names)

    all_candidates = []
    source_errors_all = []

    # Priority 1: Finnhub (professional financial news, longer lookback)
    if use_finnhub:
        finnhub_lookback = max(lookback_days, 30)  # Extend Finnhub to 30 days
        finnhub_candidates = collect_finnhub_news(
            ticker,
            now=as_of,
            lookback_days=finnhub_lookback,
            max_items=max_items * 2,
            get=get,
        )
        all_candidates.extend(finnhub_candidates)

    # Priority 2: AkShare Chinese coverage. East Money publishes few items per
    # week, so it feeds the 30-day window rather than the fresh one.
    if use_akshare:
        try:
            all_candidates.extend(
                collect_akshare_news(
                    ticker,
                    now=as_of,
                    lookback_days=max(lookback_days, 30),
                    max_items=max_items * 2,
                    fetch=akshare_fetch,
                )
            )
        except Exception as exc:  # Providers must never abort the caller.
            source_errors_all.append(type(exc).__name__)

    # Priority 3: Sina's newswire digest, the only minute-level source here.
    # It carries breaking items (product launches, buybacks) hours before the
    # slower per-symbol feeds pick them up.
    if use_sina:
        try:
            all_candidates.extend(
                collect_sina_rolling_news(
                    ticker,
                    name,
                    now=as_of,
                    lookback_days=lookback_days,
                    max_items=max_items * 2,
                    aliases=_sina_aliases(ticker, company_names),
                    get=sina_get,
                )
            )
        except Exception as exc:  # Providers must never abort the caller.
            source_errors_all.append(type(exc).__name__)

    # Priority 4: SEC EDGAR material-event filings (8-K domestic, 6-K foreign).
    # These are event-driven and cluster around earnings — a company may report
    # once a quarter, leaving 60+ day gaps. The window is extended to match the
    # cadence of the slowest low-frequency sources (Finnhub, AkShare).
    if use_sec:
        try:
            all_candidates.extend(
                collect_sec_filings(
                    ticker,
                    now=as_of,
                    lookback_days=max(lookback_days, 60),
                    max_items=max_items * 2,
                    get=sec_get,
                )
            )
        except Exception as exc:
            source_errors_all.append(type(exc).__name__)

    # Priority 5: Multi-query search (Yahoo + Google)
    for query_text in queries[:3]:  # Limit to top 3 queries to avoid rate limits
        result = collect_news(
            name="",  # Empty to use raw query
            ticker=query_text,
            now=now,
            lookback_days=lookback_days,
            fallback_days=fallback_days,
            max_items=max_items * 2,  # Collect more for merging
            search_factory=search_factory,
            get=get,
        )

        # Collect items
        for item in result.get("items", []):
            all_candidates.append({
                "title": item["title"],
                "publisher": item["publisher"],
                "published_at": item["published_at"],
                "url": item["url"],
                "summary": item["summary"],
                "source_provider": item["source_provider"],
            })

        # Collect errors
        if result.get("source_errors"):
            source_errors_all.extend(result["source_errors"])

    # Filter by relevance and apply quality scoring
    import pandas as pd

    # Build keywords for relevance filtering
    keywords = _build_relevance_keywords(name, ticker, company_names)

    # Score and filter candidates
    scored_candidates = []
    for c in all_candidates:
        # Title and body are scored separately so the gate can require a title
        # hit; ranking still uses their sum.
        title_score, body_score = _relevance_scores(
            c["title"],
            c.get("summary", ""),
            keywords
        )
        relevance_score = title_score + body_score

        # Sources that bypass the relevance gate (Finnhub, SEC, Sina) are
        # pre-attributed upstream, so a keyword miss means the title was
        # phrased generically ("Form 8-K filing", "Form 6-K filing"), not that
        # the item is irrelevant. Give them a baseline score equal to one title
        # hit so they rank competitively with keyword-matched items from
        # lower-quality sources.
        policy = _source_policy(c["source_provider"])
        if policy.bypasses_gate and relevance_score == 0:
            relevance_score = 1.0  # One title-match worth

        # Apply source quality multiplier
        quality_weight = _get_source_quality_weight(c["source_provider"])
        final_score = relevance_score * quality_weight

        # Only keep items with minimum relevance
        if _passes_relevance_gate(
            relevance_score, title_score, c["source_provider"]
        ):
            scored_candidates.append({
                "title": c["title"],
                "publisher": c["publisher"],
                "published_at": pd.Timestamp(c["published_at"]),
                "url": c["url"],
                "summary": c["summary"],
                "source_provider": c["source_provider"],
                "relevance_score": relevance_score,
                "quality_weight": quality_weight,
                "final_score": final_score,
            })

    # Sort by final score (relevance * quality) and recency
    scored_candidates.sort(
        key=lambda x: (x["final_score"], x["published_at"]),
        reverse=True
    )

    fresh_cutoff = as_of - pd.Timedelta(days=lookback_days)
    fallback_cutoff = as_of - pd.Timedelta(days=fallback_days)
    fresh = [
        item for item in scored_candidates
        if fresh_cutoff <= item["published_at"] <= as_of
    ]
    fallback = [
        item for item in scored_candidates
        if fallback_cutoff <= item["published_at"] < fresh_cutoff
    ]
    if fresh:
        selected_pool, collection_freshness, window_days = fresh, "fresh", lookback_days
    elif fallback:
        selected_pool, collection_freshness, window_days = (
            fallback,
            "low_freshness",
            fallback_days,
        )
    else:
        selected_pool, collection_freshness, window_days = [], "insufficient", lookback_days

    # Preserve the relevance/quality ordering while removing duplicate stories.
    selected = _deduplicate_ranked(selected_pool, max_items)

    # Build response
    items = [
        {
            "id": f"news-{index:02d}",
            "title": item["title"],
            "publisher": item["publisher"],
            "published_at": item["published_at"].isoformat(),
            "url": item["url"],
            "summary": item["summary"],
            "source_provider": item["source_provider"],
            "freshness": collection_freshness,
        }
        for index, item in enumerate(selected, start=1)
    ]

    status = "ok" if items else ("unavailable" if len(source_errors_all) >= 2 else "insufficient")

    return {
        "status": status,
        "name": name,
        "ticker": ticker,
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "freshness": collection_freshness,
        "items": items,
        "source_errors": list(set(source_errors_all)),
        "queries_used": queries[:3],
    }


def _deduplicate_ranked(
    candidates: list[dict[str, Any]], max_items: int
) -> list[dict[str, Any]]:
    """Deduplicate an already-ranked candidate list without reordering it."""
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        normalized_title = _dedupe_title(candidate["title"])
        canonical_url = _canonical_url(candidate["url"])
        if (
            (normalized_title and normalized_title in seen_titles)
            or (canonical_url and canonical_url in seen_urls)
        ):
            continue
        if normalized_title:
            seen_titles.add(normalized_title)
        if canonical_url:
            seen_urls.add(canonical_url)
        selected.append({**candidate, "url": canonical_url})
        if len(selected) == max_items:
            break
    return selected


def _build_relevance_keywords(name: str, ticker: str, company_names: dict) -> list[str]:
    """Build list of keywords for relevance scoring."""
    keywords = []

    # Add ticker
    if ticker:
        keywords.append(ticker.lower())
        # Remove exchange suffix for matching (e.g., "1810" from "1810.HK")
        base_ticker = ticker.split('.')[0]
        if base_ticker != ticker:
            keywords.append(base_ticker)

    # Add company names from mapping
    mapping = company_names.get(ticker, {})

    if "en" in mapping:
        keywords.extend([n.lower() for n in mapping["en"]])

    if "zh" in mapping:
        keywords.extend([n.lower() for n in mapping["zh"]])

    if "keywords" in mapping:
        keywords.extend([k.lower() for k in mapping["keywords"]])

    # Add original name
    if name:
        keywords.append(name.lower())

    return list(set(keywords))  # Remove duplicates


def _calculate_relevance_score(title: str, summary: str, keywords: list[str]) -> float:
    """
    Calculate relevance score for a news item.

    Returns:
        Score >= 3: Highly relevant (keyword in title)
        Score >= 1: Relevant (keyword in summary or partial match)
        Score < 1: Low relevance
    """
    title_score, body_score = _relevance_scores(title, summary, keywords)
    return title_score + body_score


def _relevance_scores(
    title: str, summary: str, keywords: list[str]
) -> tuple[float, float]:
    """Score title and body separately in one pass over the keywords.

    Split so the relevance gate can require a title hit without rescoring: a
    provider that emits market-wide digests names every issuer in the body, so
    only the title half proves the item is about this symbol.
    """
    if not keywords:
        return 0.5, 0.0  # Neutral score if no keywords

    import re

    title_lower = title.lower()
    summary_lower = summary.lower()

    title_score = 0.0
    body_score = 0.0

    for keyword in keywords:
        keyword_lower = keyword.lower()

        # Use word boundary matching for short keywords to avoid false positives
        # (e.g., "mi" matching "million"). CJK is excluded: it runs without
        # spaces, so \b would never match a 2-3 character company name. Short
        # CJK keywords therefore fall through to substring matching with no
        # equivalent false-positive guard — accepted because these keywords come
        # from curated company names, not free text.
        if (
            len(keyword_lower) <= 3
            and keyword_lower.isalpha()
            and keyword_lower.isascii()
        ):
            # Word boundary matching for short keywords
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            in_title = re.search(pattern, title_lower) is not None
            # Only probed when the title misses, matching the elif below.
            in_summary = (
                False if in_title else re.search(pattern, summary_lower) is not None
            )
        else:
            # Substring matching for longer keywords and special characters
            in_title = keyword_lower in title_lower
            in_summary = False if in_title else keyword_lower in summary_lower

        if in_title:
            title_score += _TITLE_MATCH_SCORE
        elif in_summary:
            body_score += _BODY_MATCH_SCORE

    return title_score, body_score


def _passes_relevance_gate(
    relevance_score: float, title_score: float, source_provider: str
) -> bool:
    """Decide whether a scored candidate is relevant enough to keep.

    Providers marked ``requires_title_match`` must earn at least one title hit;
    a body-derived score does not qualify them. See _SourcePolicy for why.
    """
    policy = _source_policy(source_provider)
    if policy.bypasses_gate:
        return True
    if policy.requires_title_match:
        return title_score >= _TITLE_MATCH_SCORE
    return relevance_score >= _BODY_MATCH_SCORE


def _get_source_quality_weight(source_provider: str) -> float:
    """Get the ranking weight multiplier for a source (see _SOURCE_POLICIES)."""
    return _source_policy(source_provider).quality_weight


def _source_policy(source_provider: str) -> _SourcePolicy:
    """Return the policy for a provider, defaulting to unweighted and ungated."""
    return _SOURCE_POLICIES.get(source_provider, _DEFAULT_SOURCE_POLICY)


def _generate_queries(name: str, ticker: str, company_names: dict) -> list[str]:
    """
    Generate multiple search queries for better coverage.

    Returns queries in priority order:
    1. English name + ticker
    2. Chinese name + ticker
    3. English name + keywords
    4. Original (fallback)
    """
    queries = []

    # Try to find mapping for this ticker
    mapping = company_names.get(ticker, {})

    # Priority 1: English names
    if "en" in mapping:
        for en_name in mapping["en"]:
            queries.append(f"{en_name} {ticker}")

    # Priority 2: Chinese names
    if "zh" in mapping:
        for zh_name in mapping["zh"]:
            queries.append(f"{zh_name} {ticker}")

    # Priority 3: Keywords
    if "keywords" in mapping and "en" in mapping:
        primary_en = mapping["en"][0]
        for keyword in mapping["keywords"][:2]:  # Top 2 keywords
            queries.append(f"{primary_en} {keyword}")

    # Priority 4: Original query as fallback
    if name and ticker:
        queries.append(f"{name} {ticker}")
    elif ticker:
        queries.append(ticker)

    return queries
