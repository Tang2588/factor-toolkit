# -*- coding: utf-8 -*-
"""把所有已跑完 Rank IC 检验的因子汇总成一份对比报告。

    python ic_report.py

读取 ``output/<因子名>/RankIC/`` 下的结果，产出：

    output/IC汇总.md                全期与分年度对比表
    output/IC汇总/IC时间序列_分因子.png   四个因子的逐月 IC
    output/IC汇总/累计IC_对比.png         四个因子的累计 IC
    output/IC汇总/年度IC_热力图.png       因子 × 年份的 IC 均值

口径：每月形成日把标准化因子对「标准化对数市值 + K−1 个行业哑变量」做截面回归，
取残差作为中性化暴露，再计算它与下一持有期收益的 Spearman 秩相关。
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

IC_DIR = "RankIC"
REPORT_DIR_NAME = "IC汇总"


def list_factors(root: Path) -> list[Path]:
    return [
        item
        for item in sorted(p for p in root.iterdir() if p.is_dir())
        if (item / IC_DIR / "ic_monthly.csv").exists()
    ]


def load_factor(factor_dir: Path) -> dict | None:
    base = factor_dir / IC_DIR
    monthly = pd.read_csv(base / "ic_monthly.csv")
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    overall = pd.read_csv(base / "ic_overall_summary.csv").iloc[0].to_dict()
    annual_path = base / "ic_annual_summary.csv"
    annual = pd.read_csv(annual_path) if annual_path.exists() else pd.DataFrame()
    return {
        "name": factor_dir.name,
        "monthly": monthly,
        "overall": overall,
        "annual": annual,
        "dir": base,
    }


def plot_ic_series(factors: list[dict], path: Path) -> None:
    """四宫格：每个因子的逐月 IC 柱状图与累计 IC 曲线。"""
    plt.rcParams["axes.unicode_minus"] = False
    count = len(factors)
    columns = 2
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.2 * rows), squeeze=False)

    for index, item in enumerate(factors):
        ax = axes[index // columns][index % columns]
        monthly = item["monthly"]
        overall = item["overall"]
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.bar(
            monthly["formation_date"],
            monthly["ic_neutral"],
            width=20,
            color=np.where(monthly["ic_neutral"] >= 0, "#176B87", "#B23A48"),
            alpha=0.85,
        )
        # 累计 IC 用右轴，便于同时看强度与累积方向
        twin = ax.twinx()
        twin.plot(
            monthly["formation_date"],
            monthly["ic_neutral"].cumsum(),
            color="#8A5A2B",
            linewidth=1.5,
        )
        twin.set_ylabel("Cumulative IC", fontsize=8, color="#8A5A2B")
        twin.tick_params(axis="y", labelsize=8, colors="#8A5A2B")
        ax.set_title(
            f"{item['name'].upper()}  IC={overall['ic_mean']:+.4f}  "
            f"IR={overall['ir']:+.3f}  t={overall['ic_t']:+.3f}  "
            f"p={overall['ic_p']:.4f}  months={int(overall['months'])}",
            fontsize=10,
        )
        ax.set_ylabel("Monthly Rank IC")
        ax.grid(axis="y", alpha=0.25)

    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    figure.suptitle("Monthly Rank IC and cumulative IC (neutralized)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_cumulative_ic(factors: list[dict], path: Path) -> None:
    """叠加图：四个因子的累计 IC 与 IC 分布。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    for index, item in enumerate(factors):
        monthly = item["monthly"]
        axes[0].plot(
            monthly["formation_date"],
            monthly["ic_neutral"].cumsum(),
            linewidth=1.6,
            color=colors[index],
            label=item["name"].upper(),
        )
    axes[0].set_ylabel("Cumulative IC")
    axes[0].set_title("Cumulative Rank IC (sum of monthly IC)")
    axes[0].legend(ncol=len(factors), fontsize=10)
    axes[0].grid(alpha=0.25)

    for index, item in enumerate(factors):
        series = item["monthly"]["ic_neutral"].to_numpy(float)
        axes[1].scatter(
            series,
            np.full(len(series), index),
            s=12,
            color=colors[index],
            alpha=0.7,
        )
    axes[1].axvline(0.0, color="#555555", linewidth=0.9)
    axes[1].axvline(0.02, color="#999999", linewidth=0.8, linestyle=":")
    axes[1].axvline(-0.02, color="#999999", linewidth=0.8, linestyle=":")
    axes[1].set_yticks(range(len(factors)))
    axes[1].set_yticklabels(
        [f"{item['name'].upper()}  (n={len(item['monthly'])})" for item in factors]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Monthly Rank IC")
    axes[1].set_title("Distribution of monthly IC (dotted lines at ±0.02)")
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_annual_heatmap(factors: list[dict], path: Path) -> None:
    """热力图：因子 × 年份的 IC 均值。"""
    plt.rcParams["axes.unicode_minus"] = False
    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    matrix = np.full((len(factors), len(years)), np.nan)
    for i, item in enumerate(factors):
        mapping = {
            int(row["period"]): float(row["ic_mean"])
            for _, row in item["annual"].iterrows()
        }
        for j, year in enumerate(years):
            if year in mapping:
                matrix[i, j] = mapping[year]

    figure, ax = plt.subplots(figsize=(1.5 * len(years) + 3, 0.9 * len(factors) + 2.5))
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 0.05
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([item["name"].upper() for item in factors])
    ax.set_title("Annual mean Rank IC")
    for i in range(len(factors)):
        for j in range(len(years)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.4f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=ax, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _f(value, digits: int = 4, kind: str = "num") -> str:
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
    if kind == "signed":
        return f"{number:+.{digits}f}"
    return f"{number:.{digits}f}"


def build_conclusions(factors: list[dict]) -> list[str]:
    """根据实际数据自动生成结论条目，避免写死的解读与数字脱节。"""
    lines: list[str] = []

    # 1. 显著性与方向
    significant = [item for item in factors if float(item["overall"]["ic_p"]) < 0.05]
    if significant:
        described = "、".join(
            f"{item['name'].upper()}（IC={_f(item['overall']['ic_mean'], 4, 'signed')}，"
            f"t={_f(item['overall']['ic_t'], 2, 'signed')}，"
            f"p={_f(item['overall']['ic_p'], 4)}）"
            for item in sorted(significant, key=lambda x: abs(float(x["overall"]["ic_t"])), reverse=True)
        )
        lines.append(f"- **5% 水平显著的因子**：{described}。")
    not_significant = [item for item in factors if float(item["overall"]["ic_p"]) >= 0.05]
    if not_significant:
        described = "、".join(
            f"{item['name'].upper()}（p={_f(item['overall']['ic_p'], 4)}）" for item in not_significant
        )
        lines.append(f"- **未达到 5% 显著**：{described}。")

    # 2. 年度集中度：年度 IC 累加值中，最强年份占多少
    for item in factors:
        annual = item["annual"]
        if annual is None or annual.empty or "ic_mean" not in annual:
            continue
        values = annual[["period", "ic_mean"]].dropna()
        if values.empty:
            continue
        total = float(values["ic_mean"].sum())
        # 用绝对值判断幅度，避免负向因子被误读成「最接近零的那年最强」
        magnitude = values["ic_mean"].abs()
        strongest = values.loc[magnitude.idxmax()]
        weakest = values.loc[magnitude.idxmin()]
        peak = float(strongest["ic_mean"])
        share = (
            f"，占年度 IC 累加的 {abs(peak) / abs(total) * 100:.0f}%"
            if total != 0 and np.sign(peak) == np.sign(total)
            else ""
        )
        lines.append(
            f"- **{item['name'].upper()} 的年度分布**：{int(strongest['period'])} 年幅度最大"
            f"（IC={_f(peak, 4, 'signed')}{share}），{int(weakest['period'])} 年幅度最小"
            f"（IC={_f(weakest['ic_mean'], 4, 'signed')}）。"
        )

    # 3. 中性化是否有帮助
    helped = [
        item
        for item in factors
        if abs(float(item["overall"]["ir"])) > abs(float(item["overall"]["ic_raw_ir"]))
    ]
    if helped:
        described = "、".join(
            f"{item['name'].upper()}（IR {_f(item['overall']['ic_raw_ir'], 3, 'signed')} → "
            f"{_f(item['overall']['ir'], 3, 'signed')}）"
            for item in helped
        )
        lines.append(f"- **中性化提升了 IR**：{described}。")
    hurt = [item for item in factors if item not in helped]
    if hurt:
        described = "、".join(
            f"{item['name'].upper()}（IR {_f(item['overall']['ic_raw_ir'], 3, 'signed')} → "
            f"{_f(item['overall']['ir'], 3, 'signed')}）"
            for item in hurt
        )
        lines.append(f"- **中性化未提升 IR**：{described}。")

    # 4. IC 波动
    if len(factors) > 1:
        volatility = sorted(factors, key=lambda x: float(x["overall"]["ic_std"]), reverse=True)
        highest, lowest = volatility[0], volatility[-1]
        lines.append(
            f"- **IC 波动**：最大 {highest['name'].upper()}（标准差 "
            f"{_f(highest['overall']['ic_std'])}），最小 {lowest['name'].upper()}（"
            f"{_f(lowest['overall']['ic_std'])}）。波动越大，IC 均值越容易被少数月份主导。"
        )

    return lines


def build_markdown(factors: list[dict], root: Path) -> str:
    lines = [
        "# 因子 Rank IC 汇总",
        "",
        "本文件由 `ic_report.py` 自动生成，汇总所有已完成 Rank IC 检验的因子。",
        "",
        "## 1. 检验口径",
        "",
        "- 每月形成日把标准化因子对「标准化对数市值 + K−1 个行业哑变量」做横截面 OLS，取残差作为**中性化暴露**；",
        "- `IC_t = Spearman(中性化暴露, 下一持有期收益)`，`IR = mean(IC) / std(IC)`；",
        "- 同时报告**未中性化**的原始 IC 作为对照，用来判断中性化是帮忙还是添乱；",
        "- 显著性用 IC 序列的独立同方差 `t` 检验，`t = IR × √月数`，不执行 Newey-West HAC；",
        "- 单月观测数低于 500 的月份跳过并记录，因此各因子的月份数可能不同。",
        "",
        "## 2. 全期结果",
        "",
        "| 因子 | 月份 | 中性化 IC 均值 | IC 标准差 | IR | IC 的 t 值 | IC 的 p 值 | IC>0 比例 | abs(IC)>0.02 比例 | 原始 IC 均值 | 原始 IR | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in factors:
        overall = item["overall"]
        lines.append(
            f"| **{item['name'].upper()}** | {int(overall['months'])} | "
            f"{_f(overall['ic_mean'], 4, 'signed')} | {_f(overall['ic_std'])} | "
            f"{_f(overall['ir'], 3, 'signed')} | {_f(overall['ic_t'], 3, 'signed')} | "
            f"{_f(overall['ic_p'])} | {_f(overall['ic_positive_ratio'], 2, 'pct')} | "
            f"{_f(overall['ic_strong_ratio'], 2, 'pct')} | "
            f"{_f(overall['ic_raw_mean'], 4, 'signed')} | {_f(overall['ic_raw_ir'], 3, 'signed')} | "
            f"{_f(overall['n_mean'], 0, 'int')} |"
        )

    lines += [
        "",
        "说明：IC 均值为中性化残差与前瞻收益的 Spearman 秩相关。**符号方向由因子自身定义决定**，"
        "例如 PB 越大越贵，IC 为负是预期之内的。IR 是月频指标、未年化；显著性用 `t = IR × √月数` 判断。",
        "",
        "## 3. 分年度 IC 均值",
        "",
    ]

    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    header = "| 因子 | " + " | ".join(str(y) for y in years) + " |"
    divider = "|---|" + "---:|" * len(years)
    lines += [header, divider]
    for item in factors:
        mapping = {
            int(row["period"]): (float(row["ic_mean"]), float(row["ir"]))
            for _, row in item["annual"].iterrows()
        }
        cells = []
        for year in years:
            if year in mapping:
                mean, ir = mapping[year]
                cells.append(f"{mean:+.4f}<br>IR={ir:+.2f}")
            else:
                cells.append("-")
        lines.append(f"| **{item['name'].upper()}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 4. 图表",
        "",
        "### 4.1 IC 时间序列（分因子）",
        "",
        "![IC时间序列](./IC汇总/IC时间序列_分因子.png)",
        "",
        "每格是该因子的逐月中性化 IC（柱）与累计 IC（棕色线，右轴）。标题给出全期 IC 均值、IR、"
        "t 值、p 值与月份数。蓝柱为正向月份、红柱为负向月份。",
        "",
        "**累计 IC 是逐月 IC 的累加，不是组合净值**，不能当作可交易收益。它反映的是 IC 的累积方向。",
        "",
        "### 4.2 累计 IC 与分布对比",
        "",
        "![累计IC](./IC汇总/累计IC_对比.png)",
        "",
        "上半部分是四个因子的累计 IC 叠加对比；下半部分是各因子月度 IC 的分布，虚线标出 ±0.02 的位置。",
        "分布越集中在零附近，说明因子的排序预测力越弱。",
        "",
        "### 4.3 年度 IC 热力图",
        "",
        "![年度IC](./IC汇总/年度IC_热力图.png)",
        "",
        "行是因子、列是年份，格内是该年的 IC 均值。绿色为正向、红色为负向。用来判断预测能力"
        "是集中在少数年份，还是分布较均匀。",
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
        "每个目录里都有逐月 IC、中性化残差样本、年度与全期汇总、图表、配置记录，"
        "以及该因子单独的 `IC统计结果.md`。",
        "",
        "## 6. 结论",
        "",
        "以下条目由实际数据自动生成，改数据后重新运行即可同步。",
        "",
    ]
    lines += build_conclusions(factors)
    lines += [
        "",
        "## 7. 怎么读这些数",
        "",
        "1. **先看符号是否符合经济含义**。负 IC 不一定是错误，可能只是因子方向本身是反的；",
        "2. **不要只看 IC 均值，要配 IR 一起看**。同样 IC 均值 0.03，标准差 0.10（IR=0.3）和标准差 0.25"
        "（IR=0.12）是完全不同的稳定性；",
        "3. **IC 显著 ≠ 组合能挣钱**。IC 只说明排序与收益有关联，强度、可交易性和扣费后的表现"
        "要看分层回测；",
        "4. **`abs(IC) > 0.02` 的月份比例通常很高**（常见 70%~90%），门槛偏松，不宜单独用它筛因子；",
        "5. **分年度形态比全期均值更重要**。全期显著但只靠一两年贡献的因子，稳定性存疑；",
        "6. **对比「中性化 IC」与「原始 IC」**可以判断中性化是否有帮助：若中性化后 IC 均值上升、"
        "标准差下降，说明原始因子里混入了行业与市值构成的噪音；",
        "7. **与横截面回归对照**：IC 显著而回归不显著，说明因子的信息在排序里而非线性幅度里，"
        "应优先用分组方式使用它。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总所有因子的 Rank IC 结果。")
    parser.add_argument("--root", default=None, help=f"结果根目录，默认 {OUTPUT_DIR}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root) if args.root else OUTPUT_DIR
    factor_dirs = list_factors(root)
    if not factor_dirs:
        raise SystemExit(f"在 {root} 下没有找到任何已完成的 Rank IC 结果")

    report_dir = root / REPORT_DIR_NAME
    report_dir.mkdir(parents=True, exist_ok=True)
    factors = [item for item in (load_factor(d) for d in factor_dirs) if item]
    print(f"因子数：{len(factors)}  ->  {[item['name'] for item in factors]}")

    series_plot = report_dir / "IC时间序列_分因子.png"
    cumulative_plot = report_dir / "累计IC_对比.png"
    annual_plot = report_dir / "年度IC_热力图.png"
    plot_ic_series(factors, series_plot)
    plot_cumulative_ic(factors, cumulative_plot)
    plot_annual_heatmap(factors, annual_plot)
    for target in (series_plot, cumulative_plot, annual_plot):
        print(f"saved: {target}")

    markdown_path = root / "IC汇总.md"
    markdown_path.write_text(build_markdown(factors, root), encoding="utf-8")
    print(f"saved: {markdown_path}")


if __name__ == "__main__":
    main()
