import unittest
from unittest.mock import patch
import pandas as pd
import naked_k_planner as p
from tests.test_naked_k_account import account


def frame():
    return pd.DataFrame({'Open': [99.0], 'High': [101.0], 'Low': [98.0], 'Close': [100.0], 'Volume': [1000.0]},
                        index=pd.to_datetime(['2026-09-04']))


def signal():
    return {'status': 'ready', 'direction': 'up', 'strength': 'strong', 'as_of': '2026-09-04',
            'close': 100.0, 'indicators': {'ema20': 99.0, 'ema50': 95.0, 'ema200': 80.0, 'atr': 3.0},
            'zones': {'nearest_support': None, 'nearest_resistance': None, 'anchored_vwap': None},
            'breakout': True, 'entry_candidate': True, 'exit_signal': False, 'entry_reference': 100.0,
            'stop_loss': 95.0, 'max_entry': 101.5, 'resistance': None, 'reasons': [],
            'validation_status': 'UNVALIDATED'}


class PlannerTests(unittest.TestCase):
    def test_unknown_account_is_explicit_and_signal_is_only_a_plan(self):
        with patch.object(p.naked_k_trend, 'analyze_trend', return_value=signal()):
            report = p.build_trade_plan('TEST', 'TEST', frame())
        self.assertEqual(report.schema_version, 'technical-trend-v1')
        self.assertEqual(report.signal_state, 'planned_long')
        self.assertEqual(report.account['status'], 'unknown')
        self.assertEqual(report.management['holding_status'], 'unknown')
        self.assertIsNone(report.target_price)
        self.assertGreater(report.risk_plan['suggested_gross_pct'], 0)
        self.assertFalse(hasattr(report, 'smart_money_signals'))
        self.assertFalse(hasattr(report, 'ai_assistant'))

    def test_drawdown_or_stale_account_blocks_new_plan(self):
        for state in [account(current_drawdown_pct=9), account(as_of='2026-09-03')]:
            with patch.object(p.naked_k_trend, 'analyze_trend', return_value=signal()):
                report = p.build_trade_plan('TEST', 'TEST', frame(), account_state=state)
            self.assertEqual(report.risk_plan['suggested_gross_pct'], 0)
            self.assertNotEqual(report.signal_state, 'planned_long')

    def test_existing_long_exits_without_opening_a_short_or_loosening_stop(self):
        s = signal(); s.update(direction='down', entry_candidate=False, exit_signal=True, stop_loss=90)
        with patch.object(p.naked_k_trend, 'analyze_trend', return_value=s):
            state = account(); state['positions'][0]['stop_loss'] = 99
            report = p.build_trade_plan('腾讯', '0700.HK', frame(), account_state=state)
        self.assertEqual(report.management['holding_status'], 'held')
        self.assertTrue(report.management['exit_next_open'])
        self.assertEqual(report.management['suggested_stop'], 99)
        self.assertEqual(report.risk_plan['suggested_gross_pct'], 0)
        self.assertNotIn('short', report.signal_state)

    def test_proposed_budgets_share_portfolio_capacity(self):
        with patch.object(p.naked_k_trend, 'analyze_trend', return_value=signal()):
            reports = [p.build_trade_plan(x, x, frame()) for x in ['AAA', 'BBB', 'CCC', 'DDD']]
        p.apply_portfolio_limits(reports)
        self.assertLessEqual(sum(r.risk_plan['suggested_gross_pct'] for r in reports), 40)


class HeldStopTests(unittest.TestCase):
    def test_existing_stop_above_close_requires_action_without_inventing_fill(self):
        state = account(positions=[{'ticker': 'TEST', 'gross_pct': 20, 'account_risk_pct': 1, 'stop_loss': 101}])
        with patch.object(p.naked_k_trend, 'analyze_trend', return_value=signal()):
            report = p.build_trade_plan('TEST', 'TEST', frame(), account_state=state)
        self.assertEqual(report.signal_state, 'exit_pending')
        self.assertTrue(report.management['stop_breached'])
        self.assertTrue(report.management['exit_next_open'])
        self.assertIsNone(report.management['suggested_stop'])
        self.assertEqual(report.management['holding_status'], 'held')
        self.assertEqual(report.risk_plan['suggested_gross_pct'], 0)
