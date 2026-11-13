# riskrule-eval

银行风控规则策略的收益与风险量化评估工具。给它一份带结果标签的历史数据和一套规则，它算出每条规则、每套策略值多少钱、风险在哪里。

覆盖六个风险管理场景，每个场景有自己的数据上传模板、收益成本口径和示例规则，共用同一套评估引擎。

| 场景 | 代码 | 数据粒度 | 标签（=1） | 模板字段数 | 示例规则数 |
|---|---|---|---|---|---|
| 反洗钱 | `aml` | 一笔被监测的交易 | 甄别后确认可疑并上报 | 18 | 7 |
| 反电诈 | `anti_telecom_fraud` | 一笔转账或支付交易 | 确认为电信网络诈骗 | 19 | 7 |
| 贷前反欺诈 | `pre_loan_fraud` | 一笔贷款申请 | 确认欺诈 | 17 | 7 |
| 贷前审批 | `pre_loan_approval` | 一笔贷款申请 | 表现期内 M3+ 坏账 | 16 | 6 |
| 贷中调额 | `credit_limit` | 一个存量账户的检视日快照 | 未来 12 个月 M3+ | 17 | 6 |
| 贷后预警 | `post_loan_warning` | 一笔存量贷款的观察日快照 | 未来 6 个月 M3+ | 17 | 7 |

## 它算什么

所有场景用同一条公式：

```
净收益 = 避免损失 − 机会成本 − 运营成本
```

- **避免损失**：规则命中且确实是正样本（可疑、欺诈、坏账、违约）带来的收益
- **机会成本**：规则命中但其实是负样本（误报、误拒、误伤）造成的损失
- **运营成本**：每条命中都要付出的处理成本（人工甄别、外呼核实、通知）

三项在每个场景里的含义不同，由场景的“取值模型”定义：

| 场景 | 避免损失 | 机会成本 | 运营成本 |
|---|---|---|---|
| 反洗钱 | 单笔有效预警的合规价值 + 交易金额 × 资金拦截比例 | 单笔误报的客户打扰成本 | 单条预警的人工甄别成本 |
| 反电诈 | 交易金额 − 已追回金额 | 误拦截体验成本 + 交易金额 × 手续费率 | 单笔外呼核实成本 |
| 贷前反欺诈 | 申请金额 × 欺诈损失率 | 预计利润 | 单笔核查成本 |
| 贷前审批 | 预计损失金额（敞口 × 违约损失率） | 预计利润 | 单笔拒绝的运营成本 |
| 贷中调额 | 拟下调额度 × 信用转换系数 × 违约损失率 | 账户年收益 ×（收入损失比例 + 流失率） | 单户通知成本 |
| 贷后预警 | 贷款余额 × 提前处置多挽回的比例 | 月均收益 × 12 × 流失率 + 打扰成本 | 单条预警处置成本 |

在此之上输出八类分析：

1. **单规则评估**：命中率、召回率、精确率、提升度、净收益，以及规则之间的重叠
2. **策略对比**：两套策略（如现行与新策略）的全部指标和差异
3. **边际贡献**：去一法，逐条剔除规则看净收益少多少，找出冗余或拖后腿的规则
4. **Swap 分析**：策略切换后新增命中与新增放行的客群质量，并与策略对比结果对账
5. **阈值扫描**：滑动评分或任一连续字段的阈值，找净收益最高点
6. **决策矩阵**：两个阈值组合下的净收益
7. **敏感性分析**：损失、机会成本、运营成本的估计有偏差时，结论会不会翻转
8. **区分度与稳定性**：KS、AUC、洛伦兹曲线，月度命中率趋势和 PSI

## 安装

需要 Python 3.9 及以上。

```bash
git clone https://github.com/amychenhua/riskrule-eval.git
cd riskrule-eval
pip install -e .          # 开发时加 ".[dev]" 以安装 pytest
```

依赖：pandas、numpy、openpyxl、pyyaml、matplotlib。

## 五分钟上手

```bash
# 用模拟数据把六个场景全部跑一遍，看看产出长什么样
riskrule demo -o demo_output
```

`demo_output/` 下会有 `templates/`（六套上传模板）、`data/`（模拟数据）、`reports/<场景>/`（Excel 报告、图表、Markdown 摘要）。仓库里的 `templates/` 和 `examples/reports/` 就是这条命令的产出。

用自己的数据：

```bash
riskrule scenarios                                   # 1. 查看场景代码
riskrule template -s pre_loan_approval -o templates  # 2. 生成该场景的上传模板（--all 生成全部）
#    ……按模板填好数据……
riskrule validate -s pre_loan_approval -d 我的数据.xlsx          # 3. 校验
riskrule init-strategy -s pre_loan_approval -o 我的策略.yaml     # 4. 导出示例策略，改成自己的规则
riskrule evaluate -s pre_loan_approval -d 我的数据.xlsx \
    --strategy 我的策略.yaml -o report                           # 5. 评估，生成报告
```

没装成命令行时，用 `python -m riskrule_eval` 代替 `riskrule`。

## 数据上传模板

每个场景一套模板，字段各不相同。`riskrule template --all` 会生成：

- `NN_<代码>_<名称>_上传模板.xlsx`：四个工作表
  - **填写须知**：数据粒度、标签定义、标签成熟所需时间、本场景的注意事项、收益成本参数
  - **数据**：只有表头，从第 2 行开始填。深蓝表头必填，浅蓝选填，橙色是标签；0/1 和枚举字段带下拉校验；鼠标停在表头上有字段说明
  - **示例数据**：5 行模拟数据，仅作格式参考
  - **字段说明**：每个字段的类型、是否必填、取值范围、含义
- `NN_<代码>_<名称>_上传模板.csv`：只有表头的 CSV
- `字段说明.md`：全部场景的字段说明（同 [docs/data-templates.md](docs/data-templates.md)）

上传时的约定：

- 支持 `.xlsx` 和 `.csv`（UTF-8 或 GBK）。Excel 读名为“数据”的工作表，没有就读第一个
- 表头用字段名或中文名都能识别，`riskrule template --header cn` 生成中文表头的模板
- 选填字段可以留空或删掉，引用它的规则会被跳过，并在报告里提示
- 可以追加模板之外的列，在自定义规则里直接按列名引用

## 策略文件

规则写成表达式，放在 YAML 里。`riskrule init-strategy` 导出的文件长这样：

```yaml
strategy:
  params:                      # 覆盖取值模型的参数
    unit_cost: 15
  rules:
    - {id: PA03, name: 多头查询5次及以上, expr: "query_cnt_3m >= 5"}
    - {id: PA04, name: 负债收入比超过0.6, expr: "dti > 0.6"}
    - {id: PA05, name: 命中黑名单, expr: "blacklist_hit == 1"}
  strategies:                  # 按顺序取前两套对比，第一套是基准
    champion:   {name: 现行策略, rules: [PA04]}
    challenger: {name: 新策略,   rules: [PA03, PA04, PA05]}
  scan:                        # 阈值扫描
    field: credit_score
    op: "<"                    # 评分越低风险越高用 <，越高风险越高用 >=
    values: {start: 500, stop: 740, step: 20}
  matrix:                      # 决策矩阵：x 条件或 y 条件任一命中
    x: {field: query_cnt_3m, op: ">=", values: [2, 3, 4, 5, 6, 7]}
    y: {field: credit_score, op: "<", values: [540, 580, 620, 660, 700]}
  monitor:
    psi_field: query_cnt_3m    # 监控哪个字段的分布
    bins: [-1, 0, 1, 2, 4, 999]   # 不写则按首月的五分位分箱
```

表达式支持：比较（`== != < <= > >=`）、`in` / `not in`、`and` / `or` / `not`、四则运算，以及 `abs` `maximum` `minimum` `isnull` `notnull` `log`。表达式由白名单语法解析器求值，不会执行任意代码。

也可以只在命令行临时改参数：`riskrule evaluate ... -p unit_cost=20 -p lgd=0.8`。

## 报告

`riskrule evaluate` 在输出目录写出：

- `<场景>_评估报告.xlsx`：结论摘要、概览与参数、单规则评估、策略对比（含 Swap）、边际贡献、阈值扫描（含决策矩阵）、敏感性分析、评分区分度、月度监控、图表
- `charts/*.png`：提升度、边际贡献、净收益瀑布图、净收益曲线、洛伦兹曲线、决策矩阵热力图
- `summary.md`：结论和主要表格

图表需要系统里有中文字体（Noto Sans CJK、微软雅黑、苹方、黑体等）。找不到时图上的文字自动改用英文。

## 在 Python 里调用

```python
from riskrule_eval import load_scenario, read_data, validate, Evaluator
from riskrule_eval.report import write_report

sc = load_scenario("post_loan_warning")
df, check = validate(read_data("我的数据.csv", sc), sc)
assert check.ok, check.to_frame()

ev = Evaluator(sc, df, params={"recovery_uplift": 0.12})
print(ev.single_rules())        # 每个方法返回一个 DataFrame
print(ev.strategies_table())
print(ev.marginal())

write_report(ev.run_all(), sc, "report", vrep=check)
```

## 使用前要知道的限制

- **金额依赖参数假设。** 合规价值、欺诈损失率、信用转换系数、提前处置挽回比例等参数的默认值只是占位，必须用本行数据校准。报告里的敏感性分析能告诉你结论对参数有多敏感。
- **选择性偏差。** 被拒绝的申请、被拦截的交易没有“如果放行会怎样”的结果。直接用已通过的样本评估会偏乐观，需要做拒绝推断，或保留一小部分对照组。本工具不做拒绝推断，只在模板的填写须知里提醒。
- **小样本。** 正样本少于 100 个时校验会给出警告，此时提升度和净收益可能随样本波动而翻转。欺诈和洗钱场景的正样本率很低，要准备足够大的样本。
- **标签要成熟。** 各场景标签成熟所需时间写在模板里（从反欺诈的 1 到 3 个月到抵押类贷款的 12 到 24 个月），未成熟的标签会低估风险。
- **贷中调额只评估降额类规则。** 提额的收益结构不同，需要另行建模。
- **模拟数据不代表真实客群。** `riskrule sample` 和 `riskrule demo` 的数据是随机生成的，示例报告里的阈值和结论不能用于真实决策。

方法的详细说明见 [docs/methodology.md](docs/methodology.md)。

## 新增或修改场景

场景定义是 `src/riskrule_eval/scenarios/` 下的 YAML 文件，包含字段、标签、取值模型和示例策略。新增场景不用改代码：

1. 复制一个现有的 YAML，改 `key`、`fields`、`value_model`、`strategy`
2. 放进 `scenarios/` 目录成为内置场景，或者放在任意位置，用路径代替场景代码：`riskrule template -s ./my_scenario.yaml`
3. 自定义场景没有内置的模拟数据生成器，模板里的“示例数据”需要在 `synth.py` 里加一个生成函数后才有

字段类型可选 `str` `int` `float` `flag`（0/1）`enum` `date` `datetime`。

## 项目结构

```
src/riskrule_eval/
  scenarios/*.yaml   六个场景的定义
  schema.py          场景与字段的数据结构、YAML 加载
  expr.py            规则表达式的安全求值器
  dataio.py          读 CSV / Excel，识别中文表头
  validate.py        按场景校验并转换类型
  engine.py          评估引擎（八类分析）
  metrics.py         KS、AUC、洛伦兹曲线、PSI
  templates.py       生成上传模板和字段说明
  charts.py          图表
  report.py          Excel 报告、Markdown 摘要、结论生成
  synth.py           模拟数据生成器
  cli.py             命令行
templates/           预先生成的六套上传模板
examples/reports/    六个场景的示例报告（模拟数据）
docs/                方法说明、字段说明
tests/               测试
```

## 开发

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_core.py` 用一份 8 行的手算数据核对引擎的每个数字，改动 `engine.py` 后先跑它。贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

MIT，见 [LICENSE](LICENSE)。本工具仅提供量化分析方法，不构成合规、风险或经营决策建议。
