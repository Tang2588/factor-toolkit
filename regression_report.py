# -*- coding: utf-8 -*-
"""把所有已跑完横截面回归的因子汇总成一份对比报告。

    python regression_report.py

读取 ``output/<因子名>/RLM_H1_独立同方差/`` 下的结果，产出：

    output/回归汇总/回归汇总.md          全期与分年度对比表
    output/回归汇总/月度系数_分因子.png   四个因子的月度系数
    output/回归汇总/累计溢价_对比.png     四个因子的累计因子溢价
    output/回归汇总/年度系数_热力图.png   因子 × 年份的系数矩阵

注意：累计溢价是月度系数的累加，**不是组合净值**，两者不能混为一谈。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUTPUT_DIR  # noqa: E402

RLM_DIR = "RLM_H1_独立同方差"
REPORT_DIR_NAME = "回归汇总"


def list_factors(root: Path) -> list[Path]:
    """返回所有已完成横截面回归的因子目录。"""
    result = []
    for factor_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        if (factor_dir / RLM_DIR / "rlm_monthly_results.csv").exists():
            result.append(factor_dir)
    return result


def load_factor(factor_dir: Path) -> dict | None:
    base = factor_dir / RLM_DIR
    monthly = pd.read_csv(base / "rlm_monthly_results.csv")
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    overall = pd.read_csv(base / "rlm_overall_summary.csv").iloc[0].to_dict()
    annual_path = base / "rlm_annual_summary.csv"
    annual = pd.read_csv(annual_path) if annual_path.exists() else pd.DataFrame()
    config_path = base / "rlm_config.json"
    return {
        "name": factor_dir.name,
        "monthly": monthly,
        "overall": overall,
        "annual": annual,
        "dir": base,
    }


def plot_monthly_coefficients(factors: list[dict], path: Path) -> None:
    """四宫格：每个因子的月度系数柱状图。"""
    plt.rcParams["axes.unicode_minus"] = False
    count = len(factors)
    columns = 2
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.0 * rows), squeeze=False)

    for index, item in enumerate(factors):
        ax = axes[index // columns][index % columns]
        monthly = item["monthly"]
        overall = item["overall"]
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.bar(
            monthly["formation_date"],
            monthly["factor_beta"] * 100,
            width=20,
            color=np.where(monthly["factor_beta"] >= 0, "#176B87", "#B23A48"),
        )
        ax.set_title(
            f"{item['name'].upper()}  mean={overall['factor_mean'] * 100:+.4f}%  "
            f"t={overall['factor_iid_t']:.3f}  p={overall['factor_iid_p']:.4f}  "
            f"months={int(overall['months'])}",
            fontsize=10,
        )
        ax.set_ylabel("Monthly coefficient (%)")
        ax.grid(axis="y", alpha=0.25)

    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.suptitle("Monthly cross-sectional Huber RLM coefficients", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_cumulative_premium(factors: list[dict], path: Path) -> None:
    """叠加图：四个因子的累计因子溢价。"""
    plt.rcParams["axes.unicode_minus"] = False
    # 上下两图的横轴含义不同（日期 vs 系数取值），不能共享坐标轴
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    for index, item in enumerate(factors):
        monthly = item["monthly"]
        axes[0].plot(
            monthly["formation_date"],
            monthly["factor_beta"].cumsum() * 100,
            linewidth=1.6,
            color=colors[index],
            label=f"{item['name'].upper()}",
        )
    axes[0].set_ylabel("Cumulative factor premium (%)")
    axes[0].set_title("Cumulative factor premium (sum of monthly coefficients, NOT portfolio NAV)")
    axes[0].legend(ncol=len(factors), fontsize=10)
    axes[0].grid(alpha=0.25)

    # 下图：把每个因子的月度系数摊开成一排散点，直观看到正负月份的分布
    for index, item in enumerate(factors):
        series = item["monthly"]["factor_beta"].to_numpy(float) * 100
        axes[1].scatter(
            series,
            np.full(len(series), index),
            s=12,
            color=colors[index],
            alpha=0.7,
        )
    axes[1].axvline(0.0, color="#555555", linewidth=0.9)
    axes[1].set_yticks(range(len(factors)))
    axes[1].set_yticklabels(
        [f"{item['name'].upper()}  (n={len(item['monthly'])})" for item in factors]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Monthly coefficient (%)")
    axes[1].set_title("Distribution of monthly coefficients")
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_annual_heatmap(factors: list[dict], path: Path) -> None:
    """热力图：因子 × 年份的年度月均系数。"""
    plt.rcParams["axes.unicode_minus"] = False
    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    matrix = np.full((len(factors), len(years)), np.nan)
    for i, item in enumerate(factors):
        mapping = {
            int(row["period"]): float(row["factor_mean"])
            for _, row in item["annual"].iterrows()
        }
        for j, year in enumerate(years):
            if year in mapping:
                matrix[i, j] = mapping[year] * 100

    figure, ax = plt.subplots(figsize=(1.4 * len(years) + 3, 0.9 * len(factors) + 2.5))
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([item["name"].upper() for item in factors])
    ax.set_title("Annual mean monthly coefficient (%)")
    for i in range(len(factors)):
        for j in range(len(years)):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:+.3f}",
                    ha="center", va="center", fontsize=9,
                )
    figure.colorbar(image, ax=ax, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _pct(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(number) else f"{number * 100:+.{digits}f}%"


def _num(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if not np.isfinite(number) else f"{number:+.{digits}f}"


def _plain(value, digits: int = 2, kind: str = "num") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if kind == "pct":
        return f"{number * 100:.{digits}f}%"
    if kind == "int":
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def build_markdown(factors: list[dict], root: Path) -> str:
    lines = [
        "# 因子横截面回归汇总（Huber RLM）",
        "",
        "本文件由 `regression_report.py` 自动生成，汇总所有已完成横截面回归的因子。",
        "",
        "## 1. 回归口径",
        "",
        "- 每月最后一个交易日形成因子，次月第一个交易日按复权收盘价建仓，再下月首个交易日退出；",
        "- 模型：`r_fwd = α + β_F · F_Z + β_Size · Size_Z + K−1 个申万一级行业哑变量 + ε`；",
        "- 估计：`HuberT(c=1.345)`、MAD 残差尺度、H1 稳健协方差，最多 100 次迭代、`1e-8` 收敛；",
        "- 单月显著性用 H1 标准误计算的大样本 `z` 值；全期显著性用月度系数的独立同方差 `t` 检验；",
        "- 单月观测数低于 500 的月份跳过并记录，因此各因子的月份数可能不同。",
        "",
        "## 2. 全期结果",
        "",
        "| 因子 | 月份 | 月均系数 | 中位数 | 标准差 | IID 标准误 | IID t 值 | IID p 值 | 正系数比例 | 单月 abs(z)>=2 比例 | 平均伪 R² | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in factors:
        overall = item["overall"]
        lines.append(
            f"| **{item['name'].upper()}** | {int(overall['months'])} | "
            f"{_pct(overall['factor_mean'])} | {_pct(overall['factor_median'])} | "
            f"{_plain(overall['factor_std'], 4, 'pct')} | {_plain(overall['factor_iid_se'], 4, 'pct')} | "
            f"{_num(overall['factor_iid_t'], 3)} | {_plain(overall['factor_iid_p'], 4)} | "
            f"{_plain(overall['factor_positive_ratio'], 2, 'pct')} | "
            f"{_plain(overall['factor_abs_monthly_z_ge_2_ratio'], 2, 'pct')} | "
            f"{_plain(overall['pseudo_r2_mean'], 2, 'pct')} | "
            f"{_plain(overall['n_mean'], 0, 'int')} |"
        )

    lines += [
        "",
        "说明：月均系数为「因子每高一个横截面标准差，下一持有期收益的平均变化」。因子已按横截面"
        "标准化，因此系数可以直接跨因子比较。**符号方向由因子自身定义决定**，例如 PB 越大越贵，"
        "系数为负是预期之内的。",
        "",
        "## 3. 分年度月均系数",
        "",
    ]

    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    header = "| 因子 | " + " | ".join(str(y) for y in years) + " |"
    divider = "|---|" + "---:|" * len(years)
    lines += [header, divider]
    for item in factors:
        mapping = {
            int(row["period"]): (float(row["factor_mean"]), float(row["factor_iid_t"]))
            for _, row in item["annual"].iterrows()
        }
        cells = []
        for year in years:
            if year in mapping:
                mean, t_value = mapping[year]
                cells.append(f"{_pct(mean, 2)}<br>t={t_value:+.2f}")
            else:
                cells.append("-")
        lines.append(f"| **{item['name'].upper()}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 4. 图表",
        "",
        "### 4.1 月度系数（分因子）",
        "",
        "![月度系数](./回归汇总/月度系数_分因子.png)",
        "",
        "每格是该因子的逐月横截面回归系数，标题给出全期均值、IID t 值与 p 值。蓝柱为正向月份、"
        "红柱为负向月份。柱子正负交替越多，说明单月估计越不稳定。",
        "",
        "### 4.2 累计因子溢价",
        "",
        "![累计溢价](./回归汇总/累计溢价_对比.png)",
        "",
        "上半部分是各因子月度系数的**累加**，下半部分是月度系数的分布（每行一个因子）。",
        "",
        "> 累计溢价是月度系数的累加和，**不是组合净值**，不能当作可交易收益。它只反映因子溢价的"
        "累积方向与稳定性。",
        "",
        "### 4.3 年度系数热力图",
        "",
        "![年度系数](./回归汇总/年度系数_热力图.png)",
        "",
        "行是因子、列是年份，格内是当年的月均系数。绿色为正向、红色为负向。这张图用来判断"
        "因子的有效性是集中在少数年份，还是分布较均匀。",
        "",
        "## 5. 各因子的完整结果",
        "",
        "| 因子 | 结果目录 |",
        "|---|---|",
    ]
    for item in factors:
        lines.append(f"| {item['name']} | `{item['dir'].relative_to(root)}` |")

    lines += [
        "",
        "每个目录里都有逐月系数、年度与全期汇总、逐股回归样本与稳健权重、样本审计表、"
        "配置记录，以及该因子单独的 `RLM统计结果.md`。",
        "",
        "## 6. 怎么读这些数",
        "",
        "1. **先看符号方向**是否符合经济含义。负号不一定是错误，可能只是因子方向本身是反的；",
        "2. **再看 IID t 值与 p 值**判断全期均值是否显著；注意这个 p 值假设月度系数独立同方差，"
        "若系数存在正自相关会被高估；",
        "3. **单月 abs(z)>=2 的比例普遍很高**（因为单月样本数大），它衡量的是单月估计精度，"
        "不是长期有效性，不要当作因子有效的证据；",
        "4. **分年度形态比全期均值更重要**。全期显著但只靠一两年贡献的因子，稳定性存疑；",
        "5. 横截面回归只回答「线性幅度」，排序层面的证据要看 Rank IC，可实现性要看分层回测。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总所有因子的横截面回归结果。")
    parser.add_argument("--root", default=None, help=f"结果根目录，默认 {OUTPUT_DIR}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root) if args.root else OUTPUT_DIR
    factor_dirs = list_factors(root)
    if not factor_dirs:
        raise SystemExit(f"在 {root} 下没有找到任何已完成的横截面回归结果")

    report_dir = root / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)

    factors = [item for item in (load_factor(d) for d in factor_dirs) if item]
    print(f"因子数：{len(factors)}  ->  {[item['name'] for item in factors]}")

    monthly_plot = report_dir / "月度系数_分因子.png"
    premium_plot = report_dir / "累计溢价_对比.png"
    annual_plot = report_dir / "年度系数_热力图.png"
    plot_monthly_coefficients(factors, monthly_plot)
    plot_cumulative_premium(factors, premium_plot)
    plot_annual_heatmap(factors, annual_plot)
    print(f"saved: {monthly_plot}")
    print(f"saved: {premium_plot}")
    print(f"saved: {annual_plot}")

    markdown_path = root / "回归汇总.md"
    markdown_path.write_text(build_markdown(factors, root), encoding="utf-8")
    print(f"saved: {markdown_path}")


if __name__ == "__main__":
    main()
