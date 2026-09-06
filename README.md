# Daily Swing Trend Analysis · 日线波段趋势分析

[最新版本 v4.0.0](https://github.com/loda13/naked-k-analysis/releases/tag/v4.0.0) · [变更记录](CHANGELOG.md) · [配置示例](config.example.json)

日线波段技术趋势评估，适合持有数周至数月。保留原入口 `naked_k_analysis.py`，使用同一套确定性规则生成报告和离线回测。**规则当前为 UNVALIDATED；工程验证不代表盈利能力已验证。**

面向A股、港股、美股等市场的研究与复盘：回答趋势方向、趋势强弱、突破是否成立、在哪里失效以及可承受多少风险。输出Markdown和JSON，支持可选账户快照；不连接券商下单。

## 核心信息

| 信息 | 用途 |
| --- | --- |
| EMA20 / 50 / 200 | 短期位置、中期趋势、长期背景 |
| ADX14 | 趋势强度；不单独判断涨跌 |
| ATR14 / ATR% | 波动、止损距离和风险预算 |
| 前20日高低通道 | 首次收盘突破及结构止损 |
| 相对成交量 | 当前量与之前20日均量比较，仅辅助观察 |
| 已确认支撑压力 / 锚定VWAP | 价格位置；VWAP为HLC3成交量近似，明确锚点和确认日 |

不再生成主力意图、双证据融合、未校准置信评分、日K分摊成交分布或AI买卖方向。新闻只保留来源、日期、标题，不调用模型。没有新增指标依赖。

## 使用

在独立Python环境中安装：

```bash
git clone https://github.com/loda13/naked-k-analysis.git
cd naked-k-analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python naked_k_analysis.py 0700.HK
```

常用命令：

```bash
python naked_k_analysis.py 0700.HK TSLA --json
python naked_k_analysis.py 0700.HK --news
python -m unittest discover -v
```

必须显式提供ticker，不内置股票池。行情依次降级：westock-data CLI、腾讯K线、Yahoo chart JSON、yfinance。默认请求3年日K和可选5年周K；周线只显示数据背景和复权检查，不重复参与交易投票。实际日线不足205根时不生成完整趋势方向或新仓计划。

`--news-lookback-days 7 --news-max-items 12` 控制事实附录，窗口外和未来消息按采集层规则过滤。标题未逐条核验正文，不等于事件已经被确认。

输出：`reports/naked_k_latest.md`、`reports/naked_k_journal.jsonl`、`reports/naked_k_audit.jsonl`。JSON及日志schema为 `technical-trend-v1`。日志追加新记录；旧记录保留原样，不自动换算为新规则的历史信号。

## 冻结的首版规则

1. 上涨：收盘 > EMA50 > EMA200，EMA50比5个交易日前高；下跌为相反关系；其余为过渡。ADX低于20显示弱、20到25为发展中、25及以上为强，仅是描述性惯例。
2. 只有上涨趋势下首次收盘突破**之前20根日K**最高价才产生候选；持续突破不反复产生新仓。EMA采用首N根收盘均值初始化；ATR/ADX采用Wilder平滑。
3. 仅紧接信号日的下一交易日开盘有效，盘中不追补。最高入场价为 `min(信号收盘 + 0.5 ATR, EMA20 + 2 ATR)`。信号收盘已经过度偏离也不生成候选。实际入场须重新检查止损距离和首个压力区下沿是否至少有1R空间。未接入节假日日历，普通工作日开盘时点之后会保守暂停旧候选，需核对下一有效交易日。
4. 初始止损为前20日最低价减0.5ATR；后续结构移动止损只收紧。收盘跌破EMA50，已有多头下一交易日开盘退出。没有固定预测目标价，不做空。
5. 计划仓位由风险预算和止损距离计算，再受单标的、市场和组合容量约束；候选按用户输入顺序分配预算。没有账户数据时这些都是假设预算，不是执行许可或真实账户暴露。

参数是预先冻结的研究规则，不是经优化得到的最优配置。ADX、成交量和VWAP不参与多项投票或概率计算；以后只有证明存在增量价值才增加过滤器。

## 账户快照

[account.example.json](account.example.json) 展示格式。复制为本地 `account.json` 后填写自己的实际数据，并更新 `as_of` 为对应行情日期；示例账户不是实际持仓。`gross_pct` 和 `account_risk_pct` 均为占账户权益的百分比，后者应包含当前持仓的风险估计；`stop_loss` 为已设保护价，也可以为null。当前只支持无杠杆多头持仓。

```bash
python naked_k_analysis.py 0700.HK --config-path config.example.json --account-path account.json
```

快照为用户输入，工具不连接券商核验，也不下单。日期过旧或来自未来时暂停新仓预算；缺失时显示 `unknown`。已知回撤达到阈值或账户已超限时停止新仓；连续亏损按配置减少风险。

快照仍有持仓、而收盘已触及原保护价时，会提示核实止损订单并安排退出，不假设订单已经成交。

止损只是计划，跳空、流动性与交易所限制可能造成更大实际损失。账户风险字段依赖输入质量；百分比预算不自动转换为各市场整手订单。

## 从 v3.x 迁移到 v4.0.0

- 本次为不兼容升级：报告结构与信号规则已替换。调用方应读取 `technical-trend-v1` schema，不能把v3历史信号拼接成同一策略业绩。
- `--llm`、`--llm-base-url`、`--llm-model`、`--news-model` 已删除，使用时明确报错。`--news` 改为事实附录。
- 旧配置的 `smart_money` 字段不再接受，按 `config.example.json` 迁移。旧输出中的AI/资金流/综合动作字段不再生成。
- MACD、RSI、KDJ、布林带未加入；当前组合已经分别描述方向、强度和波动，避免重复证据。
- 旧Python模块接口已移除，脚本集成请使用 [naked_k_trend.py](naked_k_trend.py)、[naked_k_planner.py](naked_k_planner.py) 和 [naked_k_backtest.py](naked_k_backtest.py)。CLI入口名保留。
- `docs/superpowers/` 下旧设计保留作历史，当前设计见 [日线波段精简设计](docs/superpowers/specs/2026-09-06-technical-trend-streamline-design.md)。

## 验证

v4.0.0发布前，260项离线测试及独立代码审查通过。覆盖指标计算、完整K线与复权处理、风险边界、新闻降级、执行顺序和日志一致性。生成的报告和账户文件保留在本地，不作为仓库源码发布。

离线回测的输入应包含完整、已收盘的OHLCV，明确费用、滑点和复权口径。使用实际下一日开盘、日内止损及多日持仓；期末未平仓按市值记录。比较同区间买入持有和更简单的EMA规则，明确仓位差异。

```bash
python naked_k_backtest.py 0700.HK --csv daily.csv --commission-bps 10 --slippage-bps 10 --output reports/backtest.json
```

CSV列为 `Date,Open,High,Low,Close,Volume`，按日期升序，至少206行才能评估第一个执行日。费用与滑点单位均为单边基点（10基点=0.1%），上述数值仅为命令示例；可选 `--benchmark-csv` 接收 `Date,Close` 基准，缺失则超额收益为null。比较策略采用相同仓位上限，但不采用主策略的结构止损和账户风险过滤，收益差异不能直接当作等风险超额收益。

`run_walk_forward_event_backtest(..., train_size=205, test_size=63, commission_bps=10, slippage_bps=10)` 提供冻结参数的非重叠测试窗口。训练段只作指标预热，不优化参数；窗口各自从现金开始，拼接结果不是一条连续持仓的可交易账户轨迹。

新指标的成本后超额收益尚未获得真实跨市场样本外验证。测试数据只验证计算和交易流程，不能作为收益宣传。交易日历/停牌/涨跌停/最小价位/整手/企业行为和点时复权差异均需在实盘使用前另行核对。
