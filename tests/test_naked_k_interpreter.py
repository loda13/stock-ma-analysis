import unittest
from types import SimpleNamespace

import naked_k_interpreter


class NakedKInterpreterTests(unittest.TestCase):
    def test_builds_professional_trader_brief_without_indicator_signals(self):
        report = SimpleNamespace(
            action="小仓试错",
            entry_trigger=105.0,
            stop_loss=99.0,
            target_price=117.0,
            reward_to_risk=2.0,
            position_size="最高约10.0%仓位",
            price_action={
                "bias": "bullish",
                "signals": ["下破20日低点收回", "放量下破收回"],
                "warnings": [],
                "volume_pressure": "承接增强",
            },
            market_structure={
                "direction": "transition",
                "latest_event": {"kind": "CHoCH", "direction": "bullish", "broken_level": 104.0},
                "last_swing_low": {"price": 98.0},
                "last_swing_high": {"price": 116.0},
            },
            market_regime={"label": "结构转换", "state": "transition", "direction": "bullish"},
            trade_setup={
                "name": "多头CHoCH反转试错",
                "direction": "long",
                "confidence_score": 6,
                "quality": "medium",
                "thesis": "下降结构被打断，空头追击失败",
            },
            price_zones={
                "nearest_support": {"label": "需求区", "lower": 97.5, "upper": 100.0},
                "nearest_resistance": {"label": "供给区", "lower": 116.0, "upper": 118.0},
            },
            risk_plan={
                "risk_level": "medium",
                "effective_account_risk_pct": 0.5,
                "target_r_multiple": 2.0,
            },
            timeframe_context={
                "alignment": "aligned_long",
                "decision_filter": "大周期方向与中周期结构支持日线多头机会，等待小周期触发确认",
            },
        )

        brief = naked_k_interpreter.build_trader_brief(report)

        self.assertIn("当前市场状态", brief)
        self.assertIn("多空力量分析", brief)
        self.assertIn("关键价格区域", brief)
        self.assertIn("可能交易路径", brief)
        self.assertIn("交易计划", brief)
        self.assertIn("风险点", brief)
        self.assertIn("空头追击失败", brief["多空力量分析"])
        self.assertNotIn("胜率", brief["交易计划"])
        self.assertIn("失效位置", brief["交易计划"])
        joined = " ".join(str(value) for value in brief.values())
        self.assertNotIn("MACD", joined)
        self.assertNotIn("RSI", joined)

    def test_does_not_invent_win_rate_from_setup_confidence(self):
        report = SimpleNamespace(
            action="观望",
            entry_trigger=105.0,
            stop_loss=99.0,
            target_price=None,
            reward_to_risk=None,
            position_size="0%（无新仓计划）",
            price_action={},
            market_structure={},
            market_regime={},
            trade_setup={"confidence_score": 99},
            price_zones={},
            risk_plan={},
            timeframe_context={},
            smart_money_signals={},
        )

        brief = naked_k_interpreter.build_trader_brief(report)

        self.assertNotIn("胜率", brief["交易计划"])
        self.assertNotIn("68%", brief["交易计划"])

    def test_long_plan_uses_downside_invalidation_wording(self):
        report = SimpleNamespace(
            action="小仓试错",
            entry_trigger=105.0,
            stop_loss=99.0,
            target_price=117.0,
            reward_to_risk=2.0,
            position_size="最高约10.0%仓位",
        )

        brief = naked_k_interpreter.build_trader_brief(report)

        self.assertIn("跌破失效位 99.0", brief["可能交易路径"][1])
        self.assertNotIn("跌破/突破", brief["可能交易路径"][1])
        self.assertIn("跌破失效位", brief["风险点"][0])

    def test_defensive_plan_uses_upside_invalidation_without_fake_target(self):
        report = SimpleNamespace(
            action="回避",
            entry_trigger=105.0,
            stop_loss=110.0,
            target_price=None,
            reward_to_risk=None,
            position_size="不建仓",
        )

        brief = naked_k_interpreter.build_trader_brief(report)

        self.assertIn("突破失效位 110.0", brief["可能交易路径"][1])
        self.assertIn("等待收盘确认再评估", brief["可能交易路径"][0])
        self.assertNotIn("暂无第一目标", brief["可能交易路径"][0])
        self.assertIn("突破失效位", brief["风险点"][0])

    def test_equal_trigger_and_stop_uses_touch_invalidation_wording(self):
        report = SimpleNamespace(
            action="观望",
            entry_trigger=105.0,
            stop_loss=105.0,
            target_price=None,
            reward_to_risk=None,
            position_size="0%（无新仓计划）",
        )

        brief = naked_k_interpreter.build_trader_brief(report)

        self.assertIn("触及失效位 105.0", brief["可能交易路径"][1])
        self.assertIn("触及失效位", brief["风险点"][0])

    def test_formatter_accepts_legacy_smart_money_brief_key(self):
        text = naked_k_interpreter.format_trader_brief({"主力行为研判": "旧版量价摘要"})

        self.assertIn("量价代理：旧版量价摘要", text)

    def test_fresh_proxy_signal_does_not_render_confidence_as_probability(self):
        text = naked_k_interpreter._format_smart_money_brief({
            "enabled": True,
            "overall_assessment": "量价代理偏多（规则强度未校准）",
            "signals": [{
                "label": "吸筹信号",
                "strength": "developing",
                "confidence": 88,
                "days_old": 2,
                "stale": False,
            }],
        })

        self.assertIn("吸筹信号", text)
        self.assertNotIn("88%", text)


if __name__ == "__main__":
    unittest.main()
