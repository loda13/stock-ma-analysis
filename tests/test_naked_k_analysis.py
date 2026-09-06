import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pandas as pd
import naked_k_analysis as a


def market_frame():
    close = [100 + i * 0.05 for i in range(240)]
    close[-1] += 0.5
    f = pd.DataFrame({'Open': [v - 0.1 for v in close], 'High': [v + 0.3 for v in close],
                      'Low': [v - 0.8 for v in close], 'Close': close, 'Volume': [1000.] * 240},
                     index=pd.bdate_range(end='2026-09-04', periods=240))
    f.attrs.update(source='fixture', adjustment='qfq')
    return f


class AnalysisTests(unittest.TestCase):
    def test_report_displays_full_price_zones_without_changing_plan(self):
        report = a.build_trade_plan('泡泡玛特', '9992.HK', market_frame())
        report.support, report.resistance = 153.5, 152.7
        for support, resistance, expected in [
            ({'lower': 151.0, 'upper': 156.0}, {'lower': 152.7, 'upper': 157.1},
             ('151.00–156.00', '152.70–157.10')),
            (None, None, ('未识别', '未识别')),
            ({'lower': 151.0, 'upper': 151.0}, None, ('151.00–151.00', '未识别')),
        ]:
            with self.subTest(expected=expected):
                report.trend['zones'].update(nearest_support=support, nearest_resistance=resistance)
                before = a.serialize_report(report)
                text = a.format_report('2026-09-06', [report], Path('unused'))
                self.assertIn(f'- 结构支撑区间：{expected[0]}', text)
                self.assertIn(f'- 结构压力区间：{expected[1]}', text)
                self.assertEqual(a.serialize_report(report), before)

    def test_offline_report_payload_and_append_only_journal(self):
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=market_frame()):
            journal, audit = Path(d) / 'journal.jsonl', Path(d) / 'audit.jsonl'
            original = '{"ticker":"OLD","legacy":true}\n'
            journal.write_text(original)
            text, reports = a.run_analysis([('测试', 'TEST')], journal, audit_path=audit)
            self.assertIn('EMA20', text)
            self.assertIn('ADX14', text)
            self.assertIn('真实持仓未知', text)
            self.assertNotIn('主力', text)
            self.assertNotIn('置信', text)
            self.assertEqual(reports[0].trend['direction'], 'up')
            self.assertTrue(journal.read_text().startswith(original))
            saved = json.loads(journal.read_text().splitlines()[-1])
            self.assertEqual(saved['schema_version'], 'technical-trend-v2')
            self.assertEqual(saved['signal_state'], reports[0].signal_state)
            self.assertEqual(saved['trend'], reports[0].trend)
            events = [json.loads(x)['event_type'] for x in audit.read_text().splitlines()]
            self.assertEqual(events[-1], 'run_completed')
            json.dumps(a.serialize_report(reports[0]), allow_nan=False)

    def test_news_failure_preserves_signal_and_facts_cannot_set_action(self):
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=market_frame()):
            path = Path(d) / 'j.jsonl'
            _, base = a.run_analysis([('测试', 'TEST')], path)
            with patch.object(a.naked_k_news_enhanced, 'collect_news_enhanced', side_effect=RuntimeError('secret')):
                text, failed = a.run_analysis([('测试', 'TEST')], path, news=True)
            self.assertEqual(base[0].trend, failed[0].trend)
            self.assertEqual(base[0].action, failed[0].action)
            self.assertNotIn('secret', text)
            payload = {'status': 'ok', 'items': [{'title': '<script>buy now</script>', 'publisher': 'X',
                       'url': 'javascript:alert(1)', 'published_at': '2026-09-04', 'action': '买入'}]}
            with patch.object(a.naked_k_news_enhanced, 'collect_news_enhanced', return_value=payload):
                text, factual = a.run_analysis([('测试', 'TEST')], path, news=True)
            self.assertEqual(factual[0].action, base[0].action)
            self.assertNotIn('<script>', text)
            self.assertNotIn('javascript:', text)
            self.assertNotIn('action', factual[0].news['items'][0])

    def test_cli_json_and_retired_options(self):
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=market_frame()):
            args = ['naked_k_analysis.py', 'TEST', '--json', '--report-path', d + '/r.md',
                    '--journal-path', d + '/j.jsonl', '--audit-path', d + '/a.jsonl']
            output = io.StringIO()
            with patch('sys.argv', args), redirect_stdout(output):
                self.assertEqual(a.main(), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload['items'][0]['schema_version'], 'technical-trend-v2')
            self.assertEqual(payload['schema_version'], 'technical-trend-v2')
            self.assertEqual(Path(d + '/r.md').read_text(), payload['report'])
        errors = io.StringIO()
        with patch('sys.argv', ['naked_k_analysis.py', 'TEST', '--llm']), redirect_stderr(errors):
            with self.assertRaises(SystemExit):
                a.parse_args()
        self.assertIn('已移除', errors.getvalue())

    def test_duplicate_tickers_and_invalid_limits_rejected_before_fetch(self):
        with patch.object(a, 'load_ohlcv') as load:
            with self.assertRaises(ValueError):
                a.run_analysis([('a', 'TEST'), ('b', 'TEST')], Path('unused'), news_lookback_days=7)
            with self.assertRaises(ValueError):
                a.run_analysis([('a', 'TEST')], Path('unused'), news_lookback_days=0)
            load.assert_not_called()

    def test_short_history_candidate_and_account_guards_survive_reporting(self):
        from tests.test_naked_k_account import account
        daily = market_frame().iloc[-55:]
        for state, expected in [(None, 'planned_long'), (account(current_drawdown_pct=9), 'watching')]:
            with self.subTest(account=state), TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=daily):
                journal, audit = Path(d) / 'j.jsonl', Path(d) / 'a.jsonl'
                text, reports = a.run_analysis([('新ETF', 'TEST')], journal, audit_path=audit,
                    account_state=state, now=pd.Timestamp('2026-09-06', tz='Asia/Shanghai'))
                report = reports[0]
                self.assertEqual(report.signal_state, expected)
                self.assertIn('短历史模式：EMA20/50；长期趋势未确认', text)
                self.assertNotIn('不足205根', text)
                self.assertIsNone(report.trend['indicators']['ema200'])
                self.assertEqual(report.trend['history_mode'], 'short')
                row = json.loads(journal.read_text())
                self.assertEqual(row['trend'], report.trend)
                events = [json.loads(x) for x in audit.read_text().splitlines()]
                self.assertEqual(events[0]['payload']['schema_version'], 'technical-trend-v2')
                planned = next(e['payload'] for e in events if e['event_type'] == 'plan_generated')
                self.assertEqual(planned['history_mode'], 'short')
                self.assertEqual(planned['daily_rows'], 55)
                if state is None:
                    self.assertEqual(report.account['status'], 'unknown')
                    self.assertIsNone(report.risk_plan['current_drawdown_pct'])

    def test_insufficient_history_report_uses_fifty_five_bar_minimum(self):
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=market_frame().iloc[-54:]):
            text, reports = a.run_analysis([('新ETF', 'TEST')], Path(d) / 'j.jsonl')
        self.assertIn('不足55根完整日K', text)
        self.assertEqual(reports[0].risk_plan['suggested_gross_pct'], 0)


class PersistenceAndFreshnessTests(unittest.TestCase):
    def test_journal_without_terminal_newline_is_not_corrupted(self):
        f = market_frame()
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=f):
            path = Path(d) / 'j.jsonl'
            path.write_text('{"ticker":"OLD"}')
            a.run_analysis([('测试', 'TEST')], path)
            rows = [json.loads(row) for row in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)

    def test_stale_daily_quote_blocks_new_budget_without_changing_historical_trend(self):
        f = market_frame()
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=f):
            text, reports = a.run_analysis([('测试', 'TEST')], Path(d) / 'j.jsonl',
                                          now=pd.Timestamp('2026-10-01', tz='Asia/Shanghai'))
        self.assertEqual(reports[0].trend['direction'], 'up')
        self.assertEqual(reports[0].risk_plan['suggested_gross_pct'], 0)
        self.assertIn('行情超过7个自然日', text)


class CandidateExpiryTests(unittest.TestCase):
    def test_passed_open_expires_budget_but_preserves_historical_trend(self):
        f = market_frame()
        with TemporaryDirectory() as d, patch.object(a, 'load_ohlcv', return_value=f):
            _, before = a.run_analysis([('测试', 'TEST')], Path(d) / 'j.jsonl',
                now=pd.Timestamp('2026-09-06', tz='America/New_York'))
            text, after = a.run_analysis([('测试', 'TEST')], Path(d) / 'j.jsonl',
                now=pd.Timestamp('2026-09-09 23:00', tz='Asia/Shanghai'))
        self.assertEqual(before[0].signal_state, 'planned_long')
        self.assertEqual(before[0].trend, after[0].trend)
        self.assertEqual(after[0].risk_plan['suggested_gross_pct'], 0)
        self.assertNotEqual(after[0].signal_state, 'planned_long')
        self.assertIn('候选开盘窗口已过或交易日历未核实', text)
