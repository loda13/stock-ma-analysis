"""Behavioral tests for the enhanced multi-source news collector."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch
import unittest

import pandas as pd

import naked_k_news_enhanced


NOW = pd.Timestamp("2026-07-20T12:00:00+08:00")


def collection_item(
    *,
    title: str,
    published_at: str,
    source_provider: str,
    url: str,
) -> dict[str, object]:
    return {
        "title": title,
        "publisher": "Test Wire",
        "published_at": published_at,
        "url": url,
        "summary": title,
        "source_provider": source_provider,
    }


@contextmanager
def only_akshare(mapping: dict[str, object], items: list[dict[str, object]]):
    """Stub every provider but AkShare, so a test sees only the items it supplies."""
    with (
        patch.object(
            naked_k_news_enhanced, "load_company_names", return_value=mapping
        ),
        patch.object(
            naked_k_news_enhanced, "collect_akshare_news", return_value=items
        ),
        patch.object(
            naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
        ),
        patch.object(
            naked_k_news_enhanced, "collect_sec_filings", return_value=[]
        ),
        patch.object(
            naked_k_news_enhanced,
            "collect_news",
            return_value={"items": [], "source_errors": []},
        ),
    ):
        yield


@contextmanager
def only_sina(mapping: dict[str, object], items: list[dict[str, object]]):
    """Stub every provider but Sina, so a test sees only the items it supplies."""
    with (
        patch.object(
            naked_k_news_enhanced, "load_company_names", return_value=mapping
        ),
        patch.object(
            naked_k_news_enhanced, "collect_sina_rolling_news", return_value=items
        ),
        patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
        patch.object(
            naked_k_news_enhanced, "collect_sec_filings", return_value=[]
        ),
        patch.object(
            naked_k_news_enhanced,
            "collect_news",
            return_value={"items": [], "source_errors": []},
        ),
    ):
        yield


class LiveNetworkAttempt(BaseException):
    """Raised when a test reaches the real internet.

    Derived from ``BaseException`` on purpose: every collector wraps its
    fetches in ``except Exception`` so one dead provider cannot abort the
    caller, and that same guard would silently swallow an ``AssertionError``
    here, turning a missed stub back into an invisible pass.
    """


class NoNetworkMixin:
    """Turn a missed provider stub into an immediate failure, not a hung run.

    ``collect_news_enhanced`` enables every provider by default, so forgetting
    one stub used to reach the live internet: the suite hung for tens of
    seconds per test on SEC's ~2MB ticker index and only failed by timeout,
    with nothing in the output naming the culprit. Patching ``requests.get``
    where the collectors resolve it makes the omission loud and instant, and
    keeps working when a new provider is added.
    """

    def setUp(self) -> None:
        super().setUp()

        def forbidden(*args: object, **kwargs: object) -> None:
            raise LiveNetworkAttempt(
                f"test made a live HTTP request to {args[0] if args else '?'}; "
                "stub the provider instead"
            )

        patcher = patch("requests.get", side_effect=forbidden)
        patcher.start()
        self.addCleanup(patcher.stop)


class EnhancedNewsCollectionTests(NoNetworkMixin, unittest.TestCase):
    def test_rejects_invalid_limits_before_calling_any_provider(self) -> None:
        for kwargs in (
            {"lookback_days": 0},
            {"fallback_days": 6, "lookback_days": 7},
            {"max_items": 0},
        ):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(naked_k_news_enhanced, "collect_news") as collect,
                patch.object(naked_k_news_enhanced, "collect_finnhub_news") as finnhub,
                patch.object(naked_k_news_enhanced, "collect_akshare_news") as akshare,
                patch.object(
                    naked_k_news_enhanced, "collect_sina_rolling_news"
                ) as sina,
                self.assertRaises(ValueError),
            ):
                naked_k_news_enhanced.collect_news_enhanced(
                    "拼多多", "PDD", **kwargs
                )
            collect.assert_not_called()
            finnhub.assert_not_called()
            akshare.assert_not_called()
            sina.assert_not_called()

    def test_quality_ranking_survives_deduplication_and_uses_supplied_clock(self) -> None:
        mapping = {
            "PDD": {
                "en": ["Pinduoduo"],
                "zh": ["拼多多"],
                "keywords": ["Temu"],
            }
        }
        finnhub_item = collection_item(
            title="PDD expands Temu operations",
            published_at="2026-07-18T08:00:00+00:00",
            source_provider="finnhub",
            url="https://finnhub.example/pdd",
        )
        newer_yahoo_item = collection_item(
            title="PDD market update",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="yahoo_finance",
            url="https://yahoo.example/pdd",
        )
        base_result = {
            "items": [newer_yahoo_item],
            "source_errors": [],
        }

        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value=mapping),
            patch.object(
                naked_k_news_enhanced,
                "collect_finnhub_news",
                return_value=[finnhub_item],
            ) as finnhub,
            patch.object(
                naked_k_news_enhanced, "collect_akshare_news", return_value=[]
            ) as akshare,
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ) as sina,
            patch.object(
                naked_k_news_enhanced, "collect_sec_filings", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value=base_result,
            ) as collect,
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "拼多多", "PDD", now=NOW, max_items=2
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["as_of"], NOW.isoformat())
        self.assertEqual(
            [item["source_provider"] for item in result["items"]],
            ["finnhub", "yahoo_finance"],
        )
        self.assertEqual([item["id"] for item in result["items"]], ["news-01", "news-02"])
        finnhub.assert_called_once_with(
            "PDD", now=NOW, lookback_days=30, max_items=4, get=None
        )
        akshare.assert_called_once_with(
            "PDD", now=NOW, lookback_days=30, max_items=4, fetch=None
        )
        sina.assert_called_once_with(
            "PDD",
            "拼多多",
            now=NOW,
            lookback_days=7,
            max_items=4,
            aliases=["拼多多", "Pinduoduo"],
            get=None,
        )
        self.assertEqual(collect.call_count, 3)
        self.assertEqual(
            [call.kwargs["ticker"] for call in collect.call_args_list],
            ["Pinduoduo PDD", "拼多多 PDD", "Pinduoduo Temu"],
        )

    def test_merge_drops_old_provider_items_when_fresh_items_exist(self) -> None:
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        fresh = collection_item(
            title="小米发布新车",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/xiaomi-fresh",
        )
        stale = collection_item(
            title="小米旧闻回顾",
            published_at="2026-07-10T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/xiaomi-stale",
        )

        with only_akshare(mapping, [fresh, stale]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False, lookback_days=7
            )

        self.assertEqual([item["title"] for item in result["items"]], ["小米发布新车"])
        self.assertEqual(result["window_days"], 7)
        self.assertEqual(result["freshness"], "fresh")

    def test_merge_labels_stale_only_items_as_low_freshness(self) -> None:
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        stale = collection_item(
            title="小米旧闻回顾",
            published_at="2026-07-10T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/xiaomi-stale",
        )

        with only_akshare(mapping, [stale]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米",
                "1810.HK",
                now=NOW,
                use_finnhub=False,
                lookback_days=7,
                fallback_days=30,
            )

        self.assertEqual(result["window_days"], 30)
        self.assertEqual(result["freshness"], "low_freshness")
        self.assertEqual(result["items"][0]["freshness"], "low_freshness")

    def test_merge_excludes_future_items_and_reports_insufficient_metadata(self) -> None:
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        future = collection_item(
            title="小米未来公告",
            published_at="2026-07-21T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/xiaomi-future",
        )

        with only_akshare(mapping, [future]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False, lookback_days=7
            )

        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["window_days"], 7)
        self.assertEqual(result["freshness"], "insufficient")
        self.assertEqual(result["items"], [])

    def test_short_keywords_use_word_boundaries(self) -> None:
        keywords = ["mi"]

        self.assertEqual(
            naked_k_news_enhanced._calculate_relevance_score(
                "Xiaomi ships one million cars", "", keywords
            ),
            0,
        )
        self.assertEqual(
            naked_k_news_enhanced._calculate_relevance_score(
                "MI launches a handset", "", keywords
            ),
            3,
        )

    def test_short_cjk_keywords_match_without_word_boundaries(self) -> None:
        """CJK runs without spaces, so \\b never matches a 2-3 char name."""
        for keyword, title in (
            ("小米", "小米集团回购股份"),
            ("腾讯", "腾讯控股连续2日回购"),
            ("拼多多", "拼多多旗下Temu被罚"),
        ):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    naked_k_news_enhanced._calculate_relevance_score(
                        title, "", [keyword]
                    ),
                    3,
                )

    def test_short_cjk_keywords_still_miss_unrelated_titles(self) -> None:
        self.assertEqual(
            naked_k_news_enhanced._calculate_relevance_score(
                "港股通（深）净卖出50.77亿港元", "", ["小米"]
            ),
            0,
        )

    def test_irrelevant_search_results_are_filtered(self) -> None:
        base_result = {
            "items": [
                collection_item(
                    title="Unrelated sports result",
                    published_at="2026-07-19T08:00:00+00:00",
                    source_provider="google_news_rss",
                    url="https://example.com/sports",
                )
            ],
            "source_errors": [],
        }
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(naked_k_news_enhanced, "collect_news", return_value=base_result),
            patch.object(
                naked_k_news_enhanced, "collect_akshare_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sec_filings", return_value=[]
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "测试公司", "TEST", now=NOW, use_finnhub=False
            )

        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["items"], [])

    def test_akshare_outranks_google_but_not_finnhub(self) -> None:
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        published_at = "2026-07-19T08:00:00+00:00"
        akshare_item = collection_item(
            title="小米集团回购股份",
            published_at=published_at,
            source_provider="akshare_em",
            url="https://akshare.example/xiaomi",
        )
        finnhub_item = collection_item(
            title="Xiaomi buys back shares",
            published_at=published_at,
            source_provider="finnhub",
            url="https://finnhub.example/xiaomi",
        )
        google_item = collection_item(
            title="Xiaomi in the press",
            published_at=published_at,
            source_provider="google_news_rss",
            url="https://google.example/xiaomi",
        )

        with (
            patch.object(
                naked_k_news_enhanced, "load_company_names", return_value=mapping
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_finnhub_news",
                return_value=[finnhub_item],
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_akshare_news",
                return_value=[akshare_item],
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [google_item], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, max_items=3
            )

        self.assertEqual(
            [item["source_provider"] for item in result["items"]],
            ["finnhub", "akshare_em", "google_news_rss"],
        )

    def test_market_wide_flow_tables_are_filtered_from_akshare(self) -> None:
        """Flow tables list every ticker code in the body, matching the bare code."""
        mapping = {"1810.HK": {"zh": ["小米", "小米集团"], "en": ["Xiaomi"]}}
        noise_item = {
            "title": "港股通（深）净卖出50.77亿港元",
            "publisher": "证券时报网",
            "published_at": "2026-07-19T08:00:00+00:00",
            "url": "https://akshare.example/flows",
            "summary": (
                "00981 中芯国际 港股通(深) 378267.00 -137561.41 -7.74 "
                "00700 腾讯控股 港股通(深) 319273.00 56838.99 1.16 01810"
            ),
            "source_provider": "akshare_em",
        }

        with only_akshare(mapping, [noise_item]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False
            )

        self.assertEqual(result["items"], [])

    def test_flow_tables_naming_the_company_in_the_body_are_still_filtered(self) -> None:
        """Flow-table bodies carry names too, so summed body hits must not qualify."""
        mapping = {
            "0700.HK": {
                "zh": ["腾讯", "腾讯控股"],
                "en": ["Tencent"],
            }
        }
        noise_item = {
            "title": "港股通（深）净卖出50.77亿港元",
            "publisher": "证券时报网",
            "published_at": "2026-07-19T08:00:00+00:00",
            "url": "https://akshare.example/flows",
            "summary": (
                "00981 中芯国际 港股通(深) 378267.00 -137561.41 -7.74 "
                "00700 腾讯控股 港股通(深) 319273.00 56838.99 1.16"
            ),
            "source_provider": "akshare_em",
        }

        with only_akshare(mapping, [noise_item]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "腾讯", "0700.HK", now=NOW, use_finnhub=False
            )

        self.assertEqual(result["items"], [])

    def test_akshare_items_need_a_company_name_in_the_title(self) -> None:
        """A ticker that doubles as jargon (PDD) must not qualify on its own."""
        mapping = {"PDD": {"zh": ["拼多多"], "en": ["Pinduoduo"], "keywords": ["Temu"]}}
        acronym_collision = collection_item(
            title="无问芯穹发布PDD跨集群异构推理架构 单Token成本可降低37.5%",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/infra",
        )
        genuine = collection_item(
            title="拼多多旗下Temu被欧盟罚款2亿欧元",
            published_at="2026-07-18T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/pdd",
        )

        with only_akshare(mapping, [acronym_collision, genuine]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "拼多多", "PDD", now=NOW, use_finnhub=False
            )

        self.assertEqual(
            [item["title"] for item in result["items"]],
            ["拼多多旗下Temu被欧盟罚款2亿欧元"],
        )

    def test_akshare_items_are_dropped_without_any_company_name(self) -> None:
        """Without a name to match on, the title gate cannot vet flow tables."""
        item = collection_item(
            title="某只股票的公告",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="akshare_em",
            url="https://akshare.example/unknown",
        )

        with only_akshare({}, [item]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "", "600519.SS", now=NOW, use_finnhub=False
            )

        self.assertEqual(result["items"], [])

    def test_other_providers_keep_the_looser_summary_match(self) -> None:
        """The title gate is AkShare-specific; Google may still match on body."""
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        body_match = collection_item(
            title="Handset market share shifts in Q2",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="google_news_rss",
            url="https://google.example/handsets",
        )
        body_match["summary"] = "Xiaomi gained share against its rivals."

        with (
            patch.object(
                naked_k_news_enhanced, "load_company_names", return_value=mapping
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [body_match], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False, use_akshare=False
            )

        self.assertEqual(len(result["items"]), 1)

    def test_akshare_can_be_disabled(self) -> None:
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(
                naked_k_news_enhanced, "collect_akshare_news"
            ) as akshare,
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [], "source_errors": []},
            ),
        ):
            naked_k_news_enhanced.collect_news_enhanced(
                "小米",
                "1810.HK",
                now=NOW,
                use_finnhub=False,
                use_akshare=False,
            )

        akshare.assert_not_called()

    def test_sina_can_be_disabled(self) -> None:
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news"
            ) as sina,
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [], "source_errors": []},
            ),
        ):
            naked_k_news_enhanced.collect_news_enhanced(
                "小米",
                "1810.HK",
                now=NOW,
                use_finnhub=False,
                use_sina=False,
            )

        sina.assert_not_called()

    def test_sec_can_be_disabled(self) -> None:
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sec_filings"
            ) as sec,
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [], "source_errors": []},
            ),
        ):
            naked_k_news_enhanced.collect_news_enhanced(
                "拼多多",
                "PDD",
                now=NOW,
                use_finnhub=False,
                use_sec=False,
            )

        sec.assert_not_called()

    def test_sec_failure_does_not_abort_collection(self) -> None:
        """An EDGAR outage must still leave the technical report renderable."""
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(naked_k_news_enhanced, "collect_finnhub_news", return_value=[]),
            patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_sec_filings",
                side_effect=RuntimeError("EDGAR timeout"),
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "特斯拉", "TSLA", now=NOW
            )

        # One provider failure is insufficient, not unavailable (needs >=2).
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("RuntimeError", result["source_errors"])

    def test_sina_failure_does_not_abort_collection(self) -> None:
        """A newswire outage must still leave the technical report renderable."""
        with (
            patch.object(naked_k_news_enhanced, "load_company_names", return_value={}),
            patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
            patch.object(
                naked_k_news_enhanced,
                "collect_sina_rolling_news",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False
            )

        self.assertIn("RuntimeError", result["source_errors"])

    def test_sina_items_bypass_the_relevance_gate(self) -> None:
        """Sina items are attributed by headline upstream, so they arrive scoped.

        The headline need not repeat the issuer name for the item to be real:
        "新车定价25.99万" is attributed by the collector, not by this gate.
        """
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"], "keywords": ["Mi"]}}
        sina_item = collection_item(
            title="新车定价25.99万元起",
            published_at="2026-07-19T02:00:00+00:00",
            source_provider="sina",
            url="",
        )

        with only_sina(mapping, [sina_item]):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False, max_items=3
            )

        self.assertEqual(
            [item["source_provider"] for item in result["items"]], ["sina"]
        )

    def test_sina_outranks_google_but_not_finnhub(self) -> None:
        published_at = "2026-07-19T02:00:00+00:00"
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        finnhub_item = collection_item(
            title="Xiaomi quarterly result",
            published_at=published_at,
            source_provider="finnhub",
            url="https://finnhub.example/xiaomi-q",
        )
        sina_item = collection_item(
            title="小米发布新车",
            published_at=published_at,
            source_provider="sina",
            url="",
        )
        google_item = collection_item(
            title="Xiaomi in the press",
            published_at=published_at,
            source_provider="google_news_rss",
            url="https://google.example/xiaomi-p",
        )

        with (
            patch.object(
                naked_k_news_enhanced, "load_company_names", return_value=mapping
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_finnhub_news",
                return_value=[finnhub_item],
            ),
            patch.object(naked_k_news_enhanced, "collect_akshare_news", return_value=[]),
            patch.object(
                naked_k_news_enhanced,
                "collect_sina_rolling_news",
                return_value=[sina_item],
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [google_item], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, max_items=3
            )

        self.assertEqual(
            [item["source_provider"] for item in result["items"]],
            ["finnhub", "sina", "google_news_rss"],
        )

    def test_loose_keyword_tokens_are_withheld_from_sina(self) -> None:
        """"Mi" and "盲盒" would misattribute items in a market-wide digest."""
        mapping = {
            "1810.HK": {
                "zh": ["小米", "小米集团"],
                "en": ["Xiaomi"],
                "keywords": ["Mi", "Redmi", "雷军"],
            }
        }

        self.assertEqual(
            naked_k_news_enhanced._sina_aliases("1810.HK", mapping),
            ["小米", "小米集团", "Xiaomi"],
        )

    def test_sina_aliases_are_empty_for_an_unmapped_ticker(self) -> None:
        self.assertEqual(naked_k_news_enhanced._sina_aliases("XYZ", {}), [])

    def test_akshare_failure_does_not_abort_collection(self) -> None:
        mapping = {"1810.HK": {"zh": ["小米"], "en": ["Xiaomi"]}}
        google_item = collection_item(
            title="Xiaomi ships more phones",
            published_at="2026-07-19T08:00:00+00:00",
            source_provider="google_news_rss",
            url="https://google.example/xiaomi",
        )

        def broken(*args: object, **kwargs: object) -> list[dict[str, object]]:
            raise ConnectionError("provider unavailable")

        with (
            patch.object(
                naked_k_news_enhanced, "load_company_names", return_value=mapping
            ),
            patch.object(
                naked_k_news_enhanced, "collect_akshare_news", side_effect=broken
            ),
            patch.object(
                naked_k_news_enhanced, "collect_sina_rolling_news", return_value=[]
            ),
            patch.object(
                naked_k_news_enhanced,
                "collect_news",
                return_value={"items": [google_item], "source_errors": []},
            ),
        ):
            result = naked_k_news_enhanced.collect_news_enhanced(
                "小米", "1810.HK", now=NOW, use_finnhub=False
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            [item["source_provider"] for item in result["items"]],
            ["google_news_rss"],
        )
        self.assertIn("ConnectionError", result["source_errors"])


if __name__ == "__main__":
    unittest.main()
