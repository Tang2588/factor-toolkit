# 因子测试工具箱

对任意股票横截面因子做三步测试：**横截面回归 → Rank IC → 行业内分层回测**。

工具箱与因子无关：只要因子文件符合下面的格式，换因子不需要改任何代码。

**标准化不属于本工具箱。** 逐日横截面 MAD 去极值与 Z 标准化属于因子计算层，
由各因子仓库的第 5 步完成——这样同一套去极值口径全项目只实现一次。

## 一、接口契约

因子计算层唯一需要交付的东西是一份 parquet，且必须**已经做过**逐日横截面 MAD 去极值与 Z 标准化：

```text
index   = ['date', 'stock_code']
columns = ['signal']
```

- `date` 为交易日，`stock_code` 为六位代码；
- `signal` 为**标准化后**的因子值，方向自定，缺失值表示当日该股票不参与；
- **文件名就是因子名**（`ep.parquet` → 因子名 `ep`），用于命名输出目录和报告标题。

四个已交付因子都符合这个格式，可以直接接入。若手上是未标准化的原始因子，
请先在因子计算层按 `ep_step5_MAD去极值_Z标准化.py` 的口径处理好再接入。

## 二、三步测试

| 步骤 | 脚本 | 回答的问题 |
|---|---|---|
| 1. 横截面回归 | `cross_section_rlm.py` | 因子每高一个标准差，收益平均变化多少 |
| 2. Rank IC | `rank_ic.py` | 因子的排序和收益排序是否一致、是否稳定 |
| 3. 分层回测 | `layered_backtest.py` | 真按因子建组合，收益、波动、回撤如何 |

三个步骤共用同一套样本、前瞻收益、控制变量与月度日历，结果可以直接横向对照。

## 三、快速开始

```bash
pip install -r requirements.txt

# 复制一份本机配置并改成本机的数据目录
cp config/paths.toml config/paths.local.toml

# 跑完三步，结果写到 output/<因子名>/
python run_all.py --factor ep

# 换任意因子：短名（在 factor_dirs 里找）或完整路径都行
python run_all.py --factor pb
python run_all.py --factor "D:/某处/换手率.parquet" --name 换手率

# 只跑其中几步
python run_all.py --factor ep --only ic,layered

# 把该因子的三份统计文档合并成一份并编译成 PDF
python build_report_pdf.py --factor ep
#   → output/ep/因子测试报告.md
#   → output/ep/因子测试报告.pdf

# 也可以只编译指定的一份
python build_report_pdf.py --source "output/ep/RankIC/IC统计结果.md"
```

单步也可以单独运行：

```bash
python cross_section_rlm.py --factor ep
python rank_ic.py           --factor ep
python layered_backtest.py  --factor ep
```

## 四、输出布局

所有结果写到 `output/<因子名>/` 下：

```text
output/
└── ep/
    ├── 因子测试报告.md / .pdf        三份统计文档合并而成（build_report_pdf.py 生成）
    ├── RLM_H1_独立同方差/            横截面回归
    │   ├── RLM统计结果.md            ← 单因子文档（口径 + 结果表 + 图）
    │   ├── rlm_factor_premium.png      月度系数与累计溢价
    │   ├── rlm_overall_summary.csv     全期结果
    │   ├── rlm_annual_summary.csv      分年度结果
    │   ├── rlm_coefficients_long.parquet   逐月系数长表
    │   ├── rlm_regression_sample.parquet   回归样本
    │   ├── rlm_sample_audit.csv        每月样本数与流失原因
    │   ├── rlm_config.json             本次运行口径
    │   └── ...
    ├── RankIC/                       Rank IC
    │   ├── IC统计结果.md             ← 单因子文档
    │   ├── ic_timeseries.png           月度与累计 IC
    │   ├── ic_overall_summary.csv      全期结果
    │   └── ...
    └── 分层回测/                      行业内分层回测
        ├── 分层回测结果.md            ← 单因子文档
        ├── layered_nav.png             五组与多空累计净值
        ├── layered_annual.png          分年度收益
        ├── layered_turnover.png        月度换手率
        ├── layered_performance.csv     组合绩效表
        ├── layered_holdings.parquet    逐月持仓
        └── ...
```

`因子测试报告.md` 是把下面三份统计文档合并成的单一文档：自动按「回归 → IC → 分层回测」
排序，标题降级为章节，图片路径自动改写，编译后是一份带目录的 PDF。**它不含结论文字**——
结论由使用者自行撰写，脚本不会生成也不会覆盖。

每个步骤目录里还有一份 `*统计结果.md`，它是该因子在这一步的完整文档：口径说明、
全期与分年度结果表、极值月份、诊断信息和图表。**文档里的所有数字都由脚本从结果
数据自动生成，没有人工填写的内容。**

### 该看哪些文件

| 想干什么 | 看哪个 |
|---|---|
| 单因子的回归结果 | `<因子名>/RLM_H1_独立同方差/RLM统计结果.md` |
| 单因子的 Rank IC | `<因子名>/RankIC/IC统计结果.md` |
| 单因子的分层回测 | `<因子名>/分层回测/分层回测结果.md` |
| 要一份完整的 PDF | `python build_report_pdf.py --factor <因子名>` → `output/<因子名>/因子测试报告.pdf`（三份合并成一份） |
| 追溯明细数据 | `<因子名>/.../*.csv`、`*.parquet` |
| 查本次运行口径 | `<因子名>/.../*_config.json` |

## 五、配置

`config/paths.toml`，解析优先级：环境变量 → `paths.local.toml` → `paths.toml`。

| 配置项 | 说明 |
|---|---|
| `data_root` | 基础数据目录（行情、行业分类所在处） |
| `formal_test_start` | 正式测试起点。默认 `2020-07-31` |
| `factor_dirs` | 因子文件的搜索目录，用 `--factor ep` 短名时在这里找 |
| `output_dir` | 结果输出目录，默认工具箱下的 `output/` |

## 六、开始前要知道的五件事

### 1. 因子方向不统一

四个交付因子的方向各不相同：

| 因子 | 方向 | 说明 |
|---|---|---|
| EP | 正 | 盈利收益率越高越好 |
| ROE | 正 | 净资产收益率越高越好 |
| PB | **负** | 市净率越高越贵，越大越差 |
| Size | **负** | 市值越大，未来收益越低（小盘溢价） |

工具箱**不假定方向**，只报告系数与 IC 的符号。看到负号不一定是错误——它可能只是
代表这个因子的自然方向是反的。如果要统一成「越大越好」，把 PB 换成 `BP = 1/PB`
再接入即可。

对比表里的「一致性」列会提示：回归系数、IC 均值、多空收益三者方向是否一致。
**不一致**说明排序信息与线性幅度证据冲突，需要人工判断，不宜直接使用。

### 2. 覆盖率爬坡期会被跳过

单月观测数低于最小阈值（默认 500）的月份会被**跳过并记录**，不会中断流程。
例如 ROE 在 2020 年覆盖率不足 1%，会跳过 2020-07 至 2021-02 共 8 个月，
实际分析 56 个月。跳过的月份写在 `<步骤>/skipped_months.csv`。

用 `--min-observations` 可以调整阈值。如果全部月份都被跳过，脚本会明确报错。

### 3. 测试起点是全局设置

`formal_test_start` 默认 `2020-07-31`，来自 EP 的覆盖率规则。不同因子的覆盖率
爬坡速度不同，实际生效起点由「最小观测数」进一步约束，不必为每个因子单独调。

### 4. 不能替代成交模拟

分层回测已经处理了建仓日涨停与退出日跌停的剔除，但**不包含**：成交量约束、
滑点、交易成本、以及 A 股融券的实际可行性。多空组合收益（Q5−Q1）是理论值；
实务上更可用的是「最高组相对基准的超额收益与信息比率」。

### 5. 数字口径与 EP 交付版一致

工具箱的口径与 EP 交付版完全对齐，可以直接复现 EP 的已有结果：用 EP 因子跑一遍，
会得到月均系数 +0.1623%、IC 均值 +0.0277、多空年化 +4.77%，与交付版一致。

## 七、目录结构

```text
factor-toolkit/
├── run_all.py                 一键跑完三步
├── cross_section_rlm.py       步骤一：横截面回归（Huber RLM）
├── cross_section_ols.py       步骤一对照：同口径 OLS
├── rank_ic.py                 步骤二：Rank IC
├── layered_backtest.py        步骤三：行业内分层回测
├── build_report_pdf.py        把因子的统计文档编译成 PDF
├── md_to_latex.py             Markdown → LaTeX 转换（被上一步调用）
├── common/                    共享底层
│   ├── paths.py               路径与共享常量
│   ├── parquet_io.py          分片读取
│   ├── market.py              行情清洗、B 股剔除、北交所代码映射
│   ├── calendar.py            月度形成／建仓／退出日历
│   ├── returns.py             端点口径前瞻收益
│   └── panel.py               形成日面板、行业面板、回归面板、中性化残差
├── config/paths.toml
└── requirements.txt
```

## 八、运行环境

```text
Python 3.12
numpy / pandas / pyarrow / scipy / statsmodels / matplotlib
```
