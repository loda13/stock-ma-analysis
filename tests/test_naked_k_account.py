import unittest
import naked_k_portfolio as p


def account(**changes):
    result = {'as_of': '2026-09-04', 'current_drawdown_pct': 2,
              'consecutive_losses': 1, 'positions': [
                  {'ticker': '0700.HK', 'gross_pct': 20, 'account_risk_pct': 0.5,
                   'stop_loss': 100}]}
    result.update(changes)
    return result


class AccountTests(unittest.TestCase):
    def test_unknown_and_stale_are_not_flat_or_safe(self):
        self.assertTrue(hasattr(p, 'assess_account'), 'account assessment missing')
        self.assertEqual(p.assess_account(None, '2026-09-04')['status'], 'unknown')
        self.assertEqual(p.assess_account(account(), '2026-09-07')['status'], 'stale')
        self.assertEqual(p.assess_account(account(), '2026-09-03')['status'], 'future')

    def test_actual_holdings_and_over_limits(self):
        self.assertTrue(hasattr(p, 'assess_account'), 'account assessment missing')
        result = p.assess_account(account(), '2026-09-04')
        self.assertEqual(result['status'], 'known')
        self.assertEqual(result['total_gross_pct'], 20)
        self.assertEqual(result['total_account_risk_pct'], 0.5)
        result = p.assess_account(account(current_drawdown_pct=9), '2026-09-04')
        self.assertIn('最大回撤保护', result['guardrails'])
        result = p.assess_account(account(positions=[{'ticker': 'NVDA', 'gross_pct': 35,
                                                     'account_risk_pct': 1, 'stop_loss': None}]), '2026-09-04')
        self.assertIn('NVDA单标的暴露超限', result['guardrails'])

    def test_budget_checks_do_not_round_away_small_excess(self):
        state = account(positions=[{'ticker': 'TEST', 'gross_pct': 30.004, 'account_risk_pct': 3.004}])
        result = p.assess_account(state, '2026-09-04')
        self.assertEqual(result['total_account_risk_pct'], 3.004)
        self.assertIn('TEST单标的暴露超限', result['guardrails'])
        self.assertIn('账户风险暴露超限', result['guardrails'])

    def test_malformed_or_duplicate_holdings_fail(self):
        self.assertTrue(hasattr(p, 'assess_account'), 'account assessment missing')
        for state in (account(current_drawdown_pct=float('nan')), account(consecutive_losses=-1),
                      account(positions=account()['positions'] * 2), account(as_of='yesterday'),
                      account(positions=[{'ticker': 'X', 'gross_pct': -2, 'account_risk_pct': 0}])):
            with self.subTest(state=state), self.assertRaises(ValueError):
                p.assess_account(state, '2026-09-04')


class AccountSymbolTests(unittest.TestCase):
    def test_whitespace_is_normalized_before_matching_and_uniqueness(self):
        state = account(positions=[{'ticker': ' test ', 'gross_pct': 20, 'account_risk_pct': 1}])
        result = p.assess_account(state, '2026-09-04')
        self.assertEqual(result['positions'][0]['ticker'], 'TEST')
        state['positions'].append({'ticker': 'TEST', 'gross_pct': 5, 'account_risk_pct': 0.2})
        with self.assertRaises(ValueError):
            p.assess_account(state, '2026-09-04')
