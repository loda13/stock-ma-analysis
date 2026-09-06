import unittest
import naked_k_config as c


class ConfigTests(unittest.TestCase):
    def test_risk_configuration_and_removed_modules(self):
        config = c.build_trading_config({'risk': {'account_risk_pct': 0.5}})
        self.assertEqual(config.risk.account_risk_pct, 0.5)
        self.assertFalse(hasattr(config, 'smart_money'))
        with self.assertRaisesRegex(ValueError, 'smart_money'):
            c.build_trading_config({'smart_money': {'enabled': True}})

    def test_invalid_numbers_and_unknown_fields_are_rejected(self):
        for values in ({'account_risk_pct': float('nan')}, {'account_risk_pct': -1},
                       {'max_drawdown_pct': 101}, {'consecutive_losses': 3},
                       {'consecutive_loss_limit': 1.2}, {'consecutive_loss_risk_multiplier': 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                c.build_trading_config({'risk': values})
        with self.assertRaises(ValueError):
            c.build_trading_config({'portfolio': {'max_total_gross_pct': -1}})
