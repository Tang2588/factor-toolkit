# 因子测试工具箱

对任意股票横截面因子做标准化的四步测试：**去极值标准化 → 横截面回归 → Rank IC →
行业内分层回测**。

工具箱与因子无关：只要因子文件符合下面的格式，换因子不需要改任何代码。

## 一、接口契约

因子计算层唯一需要交付的东西是一份 parquet：

```text
index   = ['date', 'stock_code']
columns = ['signal']
```

- `date` 为交易日，`stock_code` 为六位代码；
- `signal` 为因子值，方向自定，缺失值表示当日该股票不参与；
- **文件名就是因子名**（`ep.parquet` → 因子名 `ep`），用于命名输出目录和报告标题。

四个已交付因子都符合这个格式，可以直接接入。

## 二、四步测试

| 步骤 | 脚本 | 回答的问题 |
|---|---|---|
| 1. 去极值标准化 | `standardize.py` | 把原始因子处理成可比口径（仅原始因子需要） |
| 2. 横截面回归 | `cross_section_rlm.py` | 因子每高一个标准差，收益平均变化多少 |
| 3. Rank IC | `rank_ic.py` | 因子的排序和收益排序是否一致、是否稳定 |
| 4. 分层回测 | `layered_backtest.py` | 真按因子建组合，收益、波动、回撤如何 |

三个分析步骤共用同一套样本、前瞻收益、控制变量与月度日历，结果可以直接横向对照。

## 三、快速开始

```bash
pip install -r requirements.txt

# 复制一份本机配置并改成本机的数据目录
cp config/paths.toml config/paths.local.toml

# 跑完四步（因子已做过 MAD 与 Z 标准化）
python run_all.py --factor ep

# 因子还是原始值，先标准化再分析
python run_all.py --factor "D:/某处/newfac_raw.parquet" --name newfac --standardize-first

# 只跑其中几步
python run_all.py --factor ep --only ic,layered

# 汇总所有已测因子
python summarize_factors.py

# 只汇总横截面回归：全期表、分年度表、月度系数与累计溢价图
python regression_report.py
```

单步也可以单独运行：

```bash
python standardize.py      --factor "D:/某处/fac_raw.parquet" --name fac
python cross_section_rlm.py --factor ep
python rank_ic.py           --factor ep
python layered_backtest.py  --factor ep
```

## 四、输出布局

所有结果写到 `output/<因子名>/` 下：

```text
output/
├── 因子对比表.md / .csv           summarize_factors.py 生成的跨因子对比表
├── 回归汇总.md                    regression_report.py 生成的回归汇总
├── 回归汇总/                      回归汇总用的三张图
├── ep/
│   ├── standardized/             （仅 --standardize-first 时产生）
│   │   ├── ep.parquet            MAD 去极值 + Z 标准化后的因子
│   │   ├── ep_mad.parquet        仅截断、未标准化
│   │   └── processing_stats.parquet  逐日中位数、MAD、上下界、截断数
│   ├── RLM_H1_独立同方差/         横截面回归
│   ├── RankIC/                   Rank IC
│   └── 分层回测/                  行业内分层回测
└── pb/  roe/  size/ ...
```

每个步骤目录里都有一份 `*统计结果.md`，以及逐月结果、图表和 `*_config.json`
（记录本次运行的完整口径）。

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
├── run_all.py                 一键跑完四步
├── standardize.py             步骤一：MAD 去极值 + Z 标准化
├── cross_section_rlm.py       步骤二：横截面回归（Huber RLM）
├── cross_section_ols.py       步骤二对照：同口径 OLS
├── rank_ic.py                 步骤三：Rank IC
├── layered_backtest.py        步骤四：行业内分层回测
├── summarize_factors.py       跨因子对比表
├── regression_report.py       横截面回归汇总（表 + 图）
├── common/                    共享底层
│   ├── paths.py               路径与共享常量
│   ├── parquet_io.py          分片读取
│   ├── market.py              行情清洗、B 股剔除、北交所代码映射
│   ├── calendar.py            月度形成／建仓／退出日历
│   ├── returns.py             端点口径前瞻收益
│   ├── panel.py               形成日面板、行业面板、回归面板、中性化残差
│   └── streaming.py           逐日 MAD 去极值与 Z 标准化
├── config/paths.toml
└── requirements.txt
```

## 八、运行环境

```text
Python 3.12
numpy / pandas / pyarrow / scipy / statsmodels / matplotlib
```
