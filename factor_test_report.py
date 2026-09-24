# -*- coding: utf-8 -*-
"""因子测试汇总报告：把横截面回归与 Rank IC 合并成一份文档。

    python factor_test_report.py

产出：

    output/因子测试.md                       汇总文档（口径、结果、结论）
    output/因子测试图/回归_月度系数_分因子.png
    output/因子测试图/回归_累计溢价_对比.png
    output/因子测试图/回归_年度系数_热力图.png
    output/因子测试图/IC_时间序列_分因子.png
    output/因子测试图/IC_累计IC_对比.png
    output/因子测试图/IC_年度IC_热力图.png

文档中的结论由实际数据自动生成，改数据后重新运行即可同步，避免文字与数字脱节。
"""
from __future__ import annotations

import argparse
import json
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
IC_DIR = "RankIC"
LAYERED_DIR = "分层回测"
FIGURE_DIR_NAME = "因子测试图"
DOC_NAME = "因子测试.md"

GROUP_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def list_factors(root: Path) -> list[str]:
    """列出所有至少有一步测试结果的因子名。"""
    names = []
    for item in sorted(p for p in root.iterdir() if p.is_dir()):
        if (
            (item / RLM_DIR).exists()
            or (item / IC_DIR).exists()
            or (item / LAYERED_DIR).exists()
        ):
            names.append(item.name)
    return names


def load_regression(root: Path, name: str) -> dict | None:
    base = root / name / RLM_DIR
    if not (base / "rlm_monthly_results.csv").exists():
        return None
    monthly = pd.read_csv(base / "rlm_monthly_results.csv")
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    return {
        "name": name,
        "monthly": monthly,
        "overall": pd.read_csv(base / "rlm_overall_summary.csv").iloc[0].to_dict(),
        "annual": pd.read_csv(base / "rlm_annual_summary.csv")
        if (base / "rlm_annual_summary.csv").exists()
        else pd.DataFrame(),
        "dir": base,
    }


def load_ic(root: Path, name: str) -> dict | None:
    base = root / name / IC_DIR
    if not (base / "ic_monthly.csv").exists():
        return None
    monthly = pd.read_csv(base / "ic_monthly.csv")
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    return {
        "name": name,
        "monthly": monthly,
        "overall": pd.read_csv(base / "ic_overall_summary.csv").iloc[0].to_dict(),
        "annual": pd.read_csv(base / "ic_annual_summary.csv")
        if (base / "ic_annual_summary.csv").exists()
        else pd.DataFrame(),
        "dir": base,
    }


def load_layered(root: Path, name: str) -> dict | None:
    base = root / name / LAYERED_DIR
    if not (base / "layered_performance.csv").exists():
        return None
    monthly = pd.read_csv(base / "layered_monthly_returns.csv")
    monthly["formation_date"] = pd.to_datetime(monthly["formation_date"])
    monthly = monthly.sort_values("formation_date").reset_index(drop=True)
    performance = pd.read_csv(base / "layered_performance.csv")
    annual_path = base / "layered_annual_returns.csv"
    turnover_path = base / "layered_turnover.csv"
    monotonic_path = base / "layered_monotonicity.json"
    return {
        "name": name,
        "monthly": monthly,
        "performance": performance,
        "long_short": performance.iloc[-1].to_dict(),
        "annual": pd.read_csv(annual_path) if annual_path.exists() else pd.DataFrame(),
        "turnover": pd.read_csv(turnover_path) if turnover_path.exists() else pd.DataFrame(),
        "monotonic": json.loads(monotonic_path.read_text(encoding="utf-8"))
        if monotonic_path.exists()
        else {},
        "dir": base,
    }


def plot_layered_nav(factors: list[dict], path: Path) -> None:
    """四宫格：每个因子的五组累计净值与股票池基准。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = _grid(factors, 4.2)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(GROUP_LABELS)))

    for index, item in enumerate(factors):
        ax = axes[index // 2][index % 2]
        monthly = item["monthly"]
        dates = monthly["formation_date"]
        ls = item["long_short"]
        ax.axhline(1.0, color="#555555", linewidth=0.9)
        for position, label in enumerate(GROUP_LABELS):
            series = (1.0 + monthly[f"cap_{label}"].fillna(0.0)).cumprod()
            ax.plot(dates, series, linewidth=1.3, color=colors[position], label=label)
        benchmark = (1.0 + monthly["benchmark_cap"].fillna(0.0)).cumprod()
        ax.plot(dates, benchmark, linewidth=1.5, color="#333333", linestyle="--", label="Universe")
        ax.set_title(
            f"{item['name'].upper()}  Q5-Q1 ann={ls['annual_return'] * 100:+.2f}%  "
            f"Sharpe={ls['sharpe']:+.3f}  MDD={ls['max_drawdown'] * 100:.1f}%  "
            f"months={int(ls['months'])}",
            fontsize=10,
        )
        ax.set_ylabel("Cumulative NAV")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(alpha=0.25)
    for index in range(len(factors), len(axes) * 2):
        axes[index // 2][index % 2].axis("off")
    figure.suptitle("Industry-neutral layered portfolios (cap-weighted)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_long_short_nav(factors: list[dict], path: Path) -> None:
    """叠加图：四个因子的多空组合累计净值与月度收益分布。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    axes[0].axhline(1.0, color="#555555", linewidth=0.9)
    for index, item in enumerate(factors):
        monthly = item["monthly"]
        nav = (1.0 + monthly["ls_cap"].fillna(0.0)).cumprod()
        axes[0].plot(
            monthly["formation_date"], nav, linewidth=1.6, color=colors[index],
            label=f"{item['name'].upper()} (Sharpe {item['long_short']['sharpe']:+.2f})",
        )
    axes[0].set_ylabel("Cumulative NAV")
    axes[0].set_title("Long-short portfolio (Q5 - Q1, cap-weighted, before transaction cost)")
    axes[0].legend(ncol=len(factors), fontsize=10)
    axes[0].grid(alpha=0.25)

    for index, item in enumerate(factors):
        series = item["monthly"]["ls_cap"].to_numpy(float) * 100
        axes[1].scatter(series, np.full(len(series), index), s=12, color=colors[index], alpha=0.7)
    axes[1].axvline(0.0, color="#555555", linewidth=0.9)
    axes[1].set_yticks(range(len(factors)))
    axes[1].set_yticklabels(
        [
            f"{item['name'].upper()}  (win {item['long_short']['win_ratio'] * 100:.1f}%)"
            for item in factors
        ]
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Monthly long-short return (%)")
    axes[1].set_title("Distribution of monthly long-short returns")
    axes[1].grid(axis="x", alpha=0.25)

    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_long_short_heatmap(factors: list[dict], path: Path) -> None:
    """热力图：因子 × 年份的多空组合收益。"""
    plt.rcParams["axes.unicode_minus"] = False
    years = sorted({int(y) for item in factors for y in item["annual"]["year"]})
    matrix = np.full((len(factors), len(years)), np.nan)
    for i, item in enumerate(factors):
        mapping = {int(row["year"]): float(row["LS"]) for _, row in item["annual"].iterrows()}
        for j, year in enumerate(years):
            if year in mapping:
                matrix[i, j] = mapping[year] * 100

    figure, ax = plt.subplots(figsize=(1.5 * len(years) + 3, 0.9 * len(factors) + 2.5))
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 5.0
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([item["name"].upper() for item in factors])
    ax.set_title("Long-short annual return (%)")
    for i in range(len(factors)):
        for j in range(len(years)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=ax, shrink=0.8, label="%")
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _grid(factors: list[dict], height: float) -> tuple:
    rows = int(np.ceil(len(factors) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(14, height * rows), squeeze=False)
    return figure, axes


def plot_regression_coefficients(factors: list[dict], path: Path) -> None:
    """四宫格：每个因子的月度回归系数。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = _grid(factors, 4.0)
    for index, item in enumerate(factors):
        ax = axes[index // 2][index % 2]
        monthly, overall = item["monthly"], item["overall"]
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
    for index in range(len(factors), len(axes) * 2):
        axes[index // 2][index % 2].axis("off")
    figure.suptitle("Monthly cross-sectional Huber RLM coefficients", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_regression_cumulative(factors: list[dict], path: Path) -> None:
    """叠加图：四个因子的累计因子溢价与系数分布。"""
    plt.rcParams["axes.unicode_minus"] = False
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
            label=item["name"].upper(),
        )
    axes[0].set_ylabel("Cumulative factor premium (%)")
    axes[0].set_title("Cumulative factor premium (sum of monthly coefficients, NOT portfolio NAV)")
    axes[0].legend(ncol=len(factors), fontsize=10)
    axes[0].grid(alpha=0.25)

    for index, item in enumerate(factors):
        series = item["monthly"]["factor_beta"].to_numpy(float) * 100
        axes[1].scatter(series, np.full(len(series), index), s=12, color=colors[index], alpha=0.7)
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


def plot_ic_series(factors: list[dict], path: Path) -> None:
    """四宫格：每个因子的逐月 IC 与累计 IC。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = _grid(factors, 4.2)
    for index, item in enumerate(factors):
        ax = axes[index // 2][index % 2]
        monthly, overall = item["monthly"], item["overall"]
        ax.axhline(0.0, color="#555555", linewidth=0.9)
        ax.bar(
            monthly["formation_date"],
            monthly["ic_neutral"],
            width=20,
            color=np.where(monthly["ic_neutral"] >= 0, "#176B87", "#B23A48"),
            alpha=0.85,
        )
        twin = ax.twinx()
        twin.plot(
            monthly["formation_date"], monthly["ic_neutral"].cumsum(),
            color="#8A5A2B", linewidth=1.5,
        )
        twin.set_ylabel("Cumulative IC", fontsize=8, color="#8A5A2B")
        twin.tick_params(axis="y", labelsize=8, colors="#8A5A2B")
        ax.set_title(
            f"{item['name'].upper()}  IC={overall['ic_mean']:+.4f}  IR={overall['ir']:+.3f}  "
            f"t={overall['ic_t']:+.3f}  p={overall['ic_p']:.4f}  months={int(overall['months'])}",
            fontsize=10,
        )
        ax.set_ylabel("Monthly Rank IC")
        ax.grid(axis="y", alpha=0.25)
    for index in range(len(factors), len(axes) * 2):
        axes[index // 2][index % 2].axis("off")
    figure.suptitle("Monthly Rank IC and cumulative IC (neutralized)", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_ic_cumulative(factors: list[dict], path: Path) -> None:
    """叠加图：四个因子的累计 IC 与 IC 分布。"""
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    for index, item in enumerate(factors):
        monthly = item["monthly"]
        axes[0].plot(
            monthly["formation_date"], monthly["ic_neutral"].cumsum(),
            linewidth=1.6, color=colors[index], label=item["name"].upper(),
        )
    axes[0].set_ylabel("Cumulative IC")
    axes[0].set_title("Cumulative Rank IC (sum of monthly IC)")
    axes[0].legend(ncol=len(factors), fontsize=10)
    axes[0].grid(alpha=0.25)

    for index, item in enumerate(factors):
        series = item["monthly"]["ic_neutral"].to_numpy(float)
        axes[1].scatter(series, np.full(len(series), index), s=12, color=colors[index], alpha=0.7)
    for bound in (0.0, 0.02, -0.02):
        axes[1].axvline(
            bound, color="#555555" if bound == 0 else "#999999",
            linewidth=0.9 if bound == 0 else 0.8,
            linestyle="-" if bound == 0 else ":",
        )
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


def plot_heatmap(factors: list[dict], value_key: str, title: str, path: Path) -> None:
    """热力图：因子 × 年份。"""
    plt.rcParams["axes.unicode_minus"] = False
    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    matrix = np.full((len(factors), len(years)), np.nan)
    for i, item in enumerate(factors):
        mapping = {
            int(row["period"]): float(row[value_key]) for _, row in item["annual"].iterrows()
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
    ax.set_title(title)
    for i in range(len(factors)):
        for j in range(len(years)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=ax, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _num(value, digits: int = 4, kind: str = "num") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if kind == "pct":
        return f"{number * 100:+.{digits}f}%"
    if kind == "int":
        return f"{number:,.0f}"
    if kind == "signed":
        return f"{number:+.{digits}f}"
    return f"{number:.{digits}f}"


def _annual_table(factors: list[dict], value_key: str, extra_key: str | None) -> list[str]:
    years = sorted({int(y) for item in factors for y in item["annual"]["period"]})
    lines = [
        "| 因子 | " + " | ".join(str(y) for y in years) + " |",
        "|---|" + "---:|" * len(years),
    ]
    for item in factors:
        mapping = {
            int(row["period"]): row for _, row in item["annual"].iterrows()
        }
        cells = []
        for year in years:
            if year in mapping:
                row = mapping[year]
                cell = f"{float(row[value_key]):+.4f}"
                if extra_key:
                    cell += f"<br>{extra_key}={float(row[extra_key]):+.2f}"
                cells.append(cell)
            else:
                cells.append("-")
        lines.append(f"| **{item['name'].upper()}** | " + " | ".join(cells) + " |")
    return lines


def _regression_conclusions(factors: list[dict]) -> list[str]:
    lines = []
    significant = [f for f in factors if float(f["overall"]["factor_iid_p"]) < 0.05]
    if significant:
        described = "、".join(
            f"{f['name'].upper()}（{_num(f['overall']['factor_mean'], 4, 'pct')}，"
            f"t={_num(f['overall']['factor_iid_t'], 2, 'signed')}）"
            for f in sorted(significant, key=lambda x: abs(float(x["overall"]["factor_iid_t"])), reverse=True)
        )
        lines.append(f"- **回归显著（5%）**：{described}。")
    quiet = [f for f in factors if float(f["overall"]["factor_iid_p"]) >= 0.05]
    if quiet:
        described = "、".join(
            f"{f['name'].upper()}（p={_num(f['overall']['factor_iid_p'], 4)}）" for f in quiet
        )
        lines.append(f"- **回归不显著**：{described}。")

    for item in factors:
        annual = item["annual"]
        if annual is None or annual.empty:
            continue
        values = annual[["period", "factor_mean"]].dropna()
        if values.empty:
            continue
        magnitude = values["factor_mean"].abs()
        strongest = values.loc[magnitude.idxmax()]
        weakest = values.loc[magnitude.idxmin()]
        lines.append(
            f"- **{item['name'].upper()} 的年度形态**：{int(strongest['period'])} 年幅度最大"
            f"（{_num(strongest['factor_mean'], 4, 'pct')}），"
            f"{int(weakest['period'])} 年幅度最小（{_num(weakest['factor_mean'], 4, 'pct')}）。"
        )
    return lines


def _ic_conclusions(factors: list[dict]) -> list[str]:
    lines = []
    significant = [f for f in factors if float(f["overall"]["ic_p"]) < 0.05]
    if significant:
        described = "、".join(
            f"{f['name'].upper()}（IC={_num(f['overall']['ic_mean'], 4, 'signed')}，"
            f"t={_num(f['overall']['ic_t'], 2, 'signed')}）"
            for f in sorted(significant, key=lambda x: abs(float(x["overall"]["ic_t"])), reverse=True)
        )
        lines.append(f"- **IC 显著（5%）**：{described}。")
    quiet = [f for f in factors if float(f["overall"]["ic_p"]) >= 0.05]
    if quiet:
        described = "、".join(
            f"{f['name'].upper()}（p={_num(f['overall']['ic_p'], 4)}）" for f in quiet
        )
        lines.append(f"- **IC 不显著**：{described}。")

    for item in factors:
        annual = item["annual"]
        if annual is None or annual.empty:
            continue
        values = annual[["period", "ic_mean"]].dropna()
        if values.empty:
            continue
        total = float(values["ic_mean"].sum())
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
            f"- **{item['name'].upper()} 的年度 IC**：{int(strongest['period'])} 年幅度最大"
            f"（{_num(peak, 4, 'signed')}{share}），"
            f"{int(weakest['period'])} 年幅度最小（{_num(weakest['ic_mean'], 4, 'signed')}）。"
        )

    helped = [f for f in factors if abs(float(f["overall"]["ir"])) > abs(float(f["overall"]["ic_raw_ir"]))]
    if helped:
        lines.append(
            "- **中性化提升了 IR**："
            + "、".join(
                f"{f['name'].upper()}（{_num(f['overall']['ic_raw_ir'], 3, 'signed')} → "
                f"{_num(f['overall']['ir'], 3, 'signed')}）"
                for f in helped
            )
            + "。"
        )
    hurt = [f for f in factors if f not in helped]
    if hurt:
        lines.append(
            "- **中性化未提升 IR**："
            + "、".join(
                f"{f['name'].upper()}（{_num(f['overall']['ic_raw_ir'], 3, 'signed')} → "
                f"{_num(f['overall']['ir'], 3, 'signed')}）"
                for f in hurt
            )
            + "。"
        )

    if len(factors) > 1:
        ordered = sorted(factors, key=lambda x: float(x["overall"]["ic_std"]), reverse=True)
        lines.append(
            f"- **IC 波动**：最大 {ordered[0]['name'].upper()}（标准差 "
            f"{_num(ordered[0]['overall']['ic_std'])}），最小 {ordered[-1]['name'].upper()}"
            f"（{_num(ordered[-1]['overall']['ic_std'])}）。波动越大，IC 均值越容易被少数月份主导。"
        )
    return lines


def _layered_conclusions(factors: list[dict]) -> list[str]:
    lines = []
    positive = [f for f in factors if float(f["long_short"]["annual_return"]) > 0]
    negative = [f for f in factors if float(f["long_short"]["annual_return"]) <= 0]
    if positive:
        lines.append(
            "- **多空组合为正**："
            + "、".join(
                f"{f['name'].upper()}（年化 {_num(f['long_short']['annual_return'], 2, 'pct')}，"
                f"夏普 {_num(f['long_short']['sharpe'], 2, 'signed')}）"
                for f in sorted(positive, key=lambda x: -float(x["long_short"]["sharpe"]))
            )
            + "。"
        )
    if negative:
        lines.append(
            "- **多空组合为负**（方向与因子定义相反，反向后即正向）："
            + "、".join(
                f"{f['name'].upper()}（年化 {_num(f['long_short']['annual_return'], 2, 'pct')}，"
                f"夏普 {_num(f['long_short']['sharpe'], 2, 'signed')}）"
                for f in sorted(negative, key=lambda x: float(x["long_short"]["sharpe"]))
            )
            + "。"
        )

    ordered = sorted(factors, key=lambda x: abs(float(x["monotonic"]["full_period_rank_corr"])), reverse=True)
    if ordered:
        lines.append(
            "- **单调性**：组序相关系数绝对值最高 "
            f"{ordered[0]['name'].upper()}（{_num(ordered[0]['monotonic']['full_period_rank_corr'], 3, 'signed')}），"
            f"最低 {ordered[-1]['name'].upper()}（{_num(ordered[-1]['monotonic']['full_period_rank_corr'], 3, 'signed')}）。"
            "绝对值越接近 1，五组收益越单调。"
        )

    for item in factors:
        annual = item["annual"]
        if annual is None or annual.empty:
            continue
        values = annual[["year", "LS"]].dropna()
        if values.empty:
            continue
        wins = int((values["LS"] > 0).sum())
        lines.append(
            f"- **{item['name'].upper()} 的年度多空**：{len(values)} 年中 {wins} 年为正、"
            f"{len(values) - wins} 年为负。"
        )

    worst = sorted(factors, key=lambda x: float(x["long_short"]["max_drawdown"]))[0]
    lines.append(
        f"- **最大回撤最深的因子**：{worst['name'].upper()}（"
        f"{_num(worst['long_short']['max_drawdown'], 2, 'pct')}），说明组合在极端阶段承受的浮亏幅度。"
    )
    return lines


def build_markdown(
    regressions: list[dict],
    ics: list[dict],
    layered: list[dict],
    factor_names: list[str],
    root: Path,
) -> str:
    lines = [
        "# 因子测试",
        "",
        "本文档由 `factor_test_report.py` 自动生成，汇总各因子的横截面回归与 Rank IC 检验结果。",
        "文档中的表格与结论均取自实际运行数据，重新运行脚本即可同步更新。",
        "",
        f"当前纳入的因子：{'、'.join(name.upper() for name in factor_names)}。",
        "",
        "## 一、测试口径",
        "",
        "### 1.1 样本与收益口径（所有检验共用）",
        "",
        "- 每月最后一个交易日形成因子，次月第一个交易日按复权收盘价建仓，再下月首个交易日退出；",
        "- 前瞻收益为建仓日与退出日复权收盘价之比减一，任一端点停牌则置为缺失；",
        "- 股票池为沪深 A 股 + 北交所，已排除 89 只沪深 B 股；",
        "- 控制变量为「形成日横截面标准化的对数总市值」加 `K−1` 个申万一级行业哑变量；",
        "- 单月观测数低于 500 的月份跳过并记录，因此各因子的月份数可能不同。",
        "",
        "### 1.2 各检验的分工",
        "",
        "| 检验 | 回答的问题 | 强项 | 盲点 |",
        "|---|---|---|---|",
        "| 横截面回归 | 因子每高一个单位，收益平均变化多少 | 有经济单位的因子溢价 | 假设线性；受极值影响；系数显著≠能挣钱 |",
        "| Rank IC | 因子的排序和收益的排序是否一致、是否稳定 | 无量纲、可跨因子比较 | 只看排序不含强度；IC 显著≠组合收益高 |",
        "",
        "## 二、横截面回归（Huber RLM）",
        "",
        "### 2.1 回归口径",
        "",
        "- 模型：`r_fwd = α + β_F · F_Z + β_Size · Size_Z + K−1 个行业哑变量 + ε`；",
        "- 估计：`HuberT(c=1.345)`、MAD 残差尺度、H1 稳健协方差，最多 100 次迭代、`1e-8` 收敛；",
        "- 单月显著性用 H1 标准误计算的大样本 `z` 值；全期显著性用月度系数的独立同方差 `t` 检验；",
        "- 月均系数可解释为「因子每高一个横截面标准差，下一持有期收益的平均变化」。",
        "",
        "### 2.2 全期结果",
        "",
        "| 因子 | 月份 | 月均系数 | 中位数 | 标准差 | IID 标准误 | IID t 值 | IID p 值 | 正系数比例 | 单月 abs(z)>=2 比例 | 平均伪 R² | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in regressions:
        overall = item["overall"]
        lines.append(
            f"| **{item['name'].upper()}** | {int(overall['months'])} | "
            f"{_num(overall['factor_mean'], 4, 'pct')} | {_num(overall['factor_median'], 4, 'pct')} | "
            f"{_num(overall['factor_std'], 4, 'pct')} | {_num(overall['factor_iid_se'], 4, 'pct')} | "
            f"{_num(overall['factor_iid_t'], 3, 'signed')} | {_num(overall['factor_iid_p'])} | "
            f"{_num(overall['factor_positive_ratio'], 2, 'pct').lstrip('+')} | "
            f"{_num(overall['factor_abs_monthly_z_ge_2_ratio'], 2, 'pct').lstrip('+')} | "
            f"{_num(overall['pseudo_r2_mean'], 2, 'pct').lstrip('+')} | "
            f"{_num(overall['n_mean'], 0, 'int')} |"
        )

    lines += [
        "",
        "**符号方向由因子自身定义决定**：例如 PB 越大代表越贵、Size 越大代表市值越大，"
        "这两者系数为负是预期之内的。",
        "",
        "### 2.3 分年度月均系数",
        "",
    ]
    lines += _annual_table(regressions, "factor_mean", "factor_iid_t")
    lines += [
        "",
        "格式：月均系数，换行后为该年的 IID t 值。",
        "",
        "### 2.4 图",
        "",
        "![回归-月度系数](./因子测试图/回归_月度系数_分因子.png)",
        "",
        "每格是该因子的逐月横截面回归系数，蓝柱为正向月份、红柱为负向月份。柱子正负交替越多，"
        "说明单月估计越不稳定。",
        "",
        "![回归-累计溢价](./因子测试图/回归_累计溢价_对比.png)",
        "",
        "上半部分是各因子月度系数的**累加**，下半部分是月度系数的分布。",
        "",
        "> 累计溢价是月度系数的累加和，**不是组合净值**，不能当作可交易收益。",
        "",
        "![回归-年度系数](./因子测试图/回归_年度系数_热力图.png)",
        "",
        "行是因子、列是年份，格内是当年的月均系数。绿色为正向、红色为负向，用来判断有效性"
        "是否集中在少数年份。",
        "",
        "## 三、Rank IC 检验",
        "",
        "### 3.1 检验口径",
        "",
        "- 每月形成日把标准化因子对「标准化对数市值 + K−1 个行业哑变量」做横截面 OLS，"
        "取残差作为**中性化暴露**；",
        "- `IC_t = Spearman(中性化暴露, 下一持有期收益)`，`IR = mean(IC) / std(IC)`；",
        "- 同时报告**未中性化**的原始 IC 作为对照；",
        "- 显著性用 `t = IR × √月数` 判断，不执行 Newey-West HAC。",
        "",
        "### 3.2 全期结果",
        "",
        "| 因子 | 月份 | 中性化 IC 均值 | IC 标准差 | IR | IC 的 t 值 | IC 的 p 值 | IC>0 比例 | abs(IC)>0.02 比例 | 原始 IC 均值 | 原始 IR | 平均样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ics:
        overall = item["overall"]
        lines.append(
            f"| **{item['name'].upper()}** | {int(overall['months'])} | "
            f"{_num(overall['ic_mean'], 4, 'signed')} | {_num(overall['ic_std'])} | "
            f"{_num(overall['ir'], 3, 'signed')} | {_num(overall['ic_t'], 3, 'signed')} | "
            f"{_num(overall['ic_p'])} | "
            f"{_num(overall['ic_positive_ratio'], 2, 'pct').lstrip('+')} | "
            f"{_num(overall['ic_strong_ratio'], 2, 'pct').lstrip('+')} | "
            f"{_num(overall['ic_raw_mean'], 4, 'signed')} | {_num(overall['ic_raw_ir'], 3, 'signed')} | "
            f"{_num(overall['n_mean'], 0, 'int')} |"
        )

    lines += [
        "",
        "IR 是月频指标、未年化。对比「中性化 IC」与「原始 IC」可以判断中性化是否有帮助。",
        "",
        "### 3.3 分年度 IC 均值",
        "",
    ]
    lines += _annual_table(ics, "ic_mean", "ir")
    lines += [
        "",
        "格式：IC 均值，换行后为该年的 IR。",
        "",
        "### 3.4 图",
        "",
        "![IC-时间序列](./因子测试图/IC_时间序列_分因子.png)",
        "",
        "每格是该因子的逐月中性化 IC（柱）与累计 IC（棕色线，右轴）。**累计 IC 是逐月 IC 的累加，"
        "不是组合净值**。",
        "",
        "![IC-累计对比](./因子测试图/IC_累计IC_对比.png)",
        "",
        "上半部分是四个因子的累计 IC 叠加；下半部分是月度 IC 的分布，虚线标出 ±0.02 的位置。",
        "",
        "![IC-年度热力图](./因子测试图/IC_年度IC_热力图.png)",
        "",
        "行是因子、列是年份，格内是该年的 IC 均值。",
        "",
        "## 四、行业内分层回测",
        "",
        "### 4.1 回测口径",
        "",
        "- **分组依据**：因子对「对数市值 + 行业」回归后的中性化残差，与 Rank IC 口径一致；",
        "- **分组方式**：每月在每个申万一级行业内部按残差升序分五组，Q1 最低、Q5 最高；"
        "各行业同序号组合并成全市场组合，因此每个组合的行业构成与股票池一致；",
        "- **权重**：组合市值加权，权重在形成日确定，持有期内不调仓；",
        "- **可交易性**：建仓日涨停（买不进）与退出日跌停（卖不出）的股票均剔除；",
        "- **多空组合**：`Q5 − Q1`，零成本对冲；",
        "- **基准**：股票池市值加权收益；",
        "- **不含交易成本**。多空组合在 A 股还受融券可行性限制，实务上更可用的是最高组相对基准的超额。",
        "",
        "### 4.2 组合绩效",
        "",
        "| 因子 | 组合 | 年化收益 | 累计收益 | 年化波动 | 夏普比率 | 最大回撤 | 胜率 | 年化超额 | 信息比率 | 平均换手率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in layered:
        table = item["performance"]
        for position in range(len(table)):
            row = table.iloc[position]
            prefix = f"**{item['name'].upper()}**" if position == 0 else ""
            lines.append(
                f"| {prefix} | {row.iloc[0]} | {_num(row['annual_return'], 2, 'pct')} | "
                f"{_num(row['total_return'], 2, 'pct')} | "
                f"{_num(row['annual_vol'], 2, 'pct').lstrip('+')} | "
                f"{_num(row['sharpe'], 3, 'signed')} | {_num(row['max_drawdown'], 2, 'pct')} | "
                f"{_num(row['win_ratio'], 2, 'pct').lstrip('+')} | "
                f"{_num(row['excess_annual_return'], 2, 'pct')} | "
                f"{_num(row['info_ratio'], 3, 'signed')} | "
                f"{_num(row['avg_turnover'], 2, 'pct').lstrip('+')} |"
            )

    lines += [
        "",
        "「年化超额」「信息比率」是相对股票池市值加权基准的多头超额；多空组合是零成本对冲组合，"
        "与多头基准比较没有经济含义，该行只看年化收益、夏普比率与最大回撤。",
        "",
        "### 4.3 单调性检验",
        "",
        "| 因子 | 全期组序相关系数 | 逐月相关系数均值 | 逐月为正比例 |",
        "|---|---:|---:|---:|",
    ]
    for item in layered:
        lines.append(
            f"| **{item['name'].upper()}** | "
            f"{_num(item['monotonic']['full_period_rank_corr'], 3, 'signed')} | "
            f"{_num(item['monotonic']['monthly_rank_corr_mean'], 4, 'signed')} | "
            f"{_num(item['monotonic']['monthly_rank_corr_positive_ratio'], 2, 'pct').lstrip('+')} |"
        )

    lines += [
        "",
        "组序相关系数越接近 1，五组收益越单调。逐月相关系数均值通常远小于全期值，"
        "说明单月排序很乱，有效性靠时间累积。",
        "",
        "### 4.4 分年度多空收益",
        "",
    ]
    years = sorted({int(y) for item in layered for y in item["annual"]["year"]})
    lines.append("| 因子 | " + " | ".join(str(y) for y in years) + " |")
    lines.append("|---|" + "---:|" * len(years))
    for item in layered:
        mapping = {int(row["year"]): float(row["LS"]) for _, row in item["annual"].iterrows()}
        cells = [(f"{mapping[y] * 100:+.2f}%" if y in mapping else "-") for y in years]
        lines.append(f"| **{item['name'].upper()}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### 4.5 图",
        "",
        "![分层-五组净值](./因子测试图/分层_五组净值_分因子.png)",
        "",
        "每格是该因子的五个分位组合与股票池基准的累计净值。**五条线是否整齐排开**是判断单调性的"
        "直观方式；线交叉说明中间分位没有区分度。",
        "",
        "![分层-多空净值](./因子测试图/分层_多空净值_对比.png)",
        "",
        "上半部分是四个因子的多空组合累计净值叠加；下半部分是月度多空收益的分布。"
        "**净值曲线不是单调向上的因子，说明它在某些阶段是反向的。**",
        "",
        "![分层-年度多空](./因子测试图/分层_年度多空_热力图.png)",
        "",
        "行是因子、列是年份，格内是当年的多空收益，用来判断收益是集中在少数年份还是分布较均匀。",
        "",
        "## 五、三个检验的对照",
        "",
        "三个检验用的是同一套样本与收益，但角度不同：回归看数值、IC 看排序、分层回测看落到"
        "组合上的结果。**方向是否一致本身就是信息。**",
        "",
        "| 因子 | 回归月均系数 | IC 均值 | 多空年化收益 | 方向是否一致 |",
        "|---|---:|---:|---:|---|",
    ]
    regression_by_name = {item["name"]: item for item in regressions}
    layered_by_name = {item["name"]: item for item in layered}
    ic_by_name = {item["name"]: item for item in ics}
    for name in factor_names:
        regression = regression_by_name.get(name)
        ic = ic_by_name.get(name)
        layered_item = layered_by_name.get(name)
        beta_mean = float(regression["overall"]["factor_mean"]) if regression else np.nan
        ic_mean = float(ic["overall"]["ic_mean"]) if ic else np.nan
        ls_return = float(layered_item["long_short"]["annual_return"]) if layered_item else np.nan
        signs = [np.sign(v) for v in (beta_mean, ic_mean, ls_return) if np.isfinite(v)]
        verdict = (
            "数据不足" if len(signs) < 3
            else ("一致" if len(set(signs)) == 1 else "**不一致**")
        )
        lines.append(
            f"| **{name.upper()}** | {_num(beta_mean, 4, 'pct')} | {_num(ic_mean, 4, 'signed')} | "
            f"{_num(ls_return, 2, 'pct')} | {verdict} |"
        )

    lines += [
        "",
        "三个方向一致，说明「线性幅度」「排序信息」「组合可实现性」互相印证；不一致说明因子的"
        "信息分布特殊（例如只有两端有区分度），需要人工判断，不宜直接使用。",
        "",
        "## 六、结论",
        "",
        "### 6.1 横截面回归",
        "",
    ]
    lines += _regression_conclusions(regressions)
    lines += ["", "### 6.2 Rank IC", ""]
    lines += _ic_conclusions(ics)
    if layered:
        lines += ["", "### 6.3 行业内分层回测", ""]
        lines += _layered_conclusions(layered)
    lines += [
        "",
        "### 6.4 使用建议",
        "",
        "- **符号为负不一定是错误**，要先看因子的经济含义。PB 越大越贵、Size 越大代表市值越大，"
        "它们与收益负相关是正常现象；若要统一成「越大越好」，把 PB 换成 `BP = 1/PB` 再接入；",
        "- **两个检验都显著且方向一致**的因子，才值得进入下一步的行业内分层回测；",
        "- 单月 `abs(z) >= 2` 的比例普遍很高（单月样本数大所致），它衡量单月估计精度，"
        "**不是长期有效性的证据**；",
        "- 全期显著但只靠一两年贡献的因子，稳定性存疑；分年度形态比全期均值更重要。",
        "",
        "## 七、各因子的完整结果",
        "",
        "| 因子 | 横截面回归 | Rank IC | 行业内分层回测 |",
        "|---|---|---|---|",
    ]
    for name in factor_names:
        regression = regression_by_name.get(name)
        ic = next((item for item in ics if item["name"] == name), None)
        layered_item = layered_by_name.get(name)
        cells = [
            f"`{regression['dir'].relative_to(root)}`" if regression else "未运行",
            f"`{ic['dir'].relative_to(root)}`" if ic else "未运行",
            f"`{layered_item['dir'].relative_to(root)}`" if layered_item else "未运行",
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "每个目录里都有逐月结果、年度与全期汇总、审计表、配置记录，以及该步骤单独的统计报告。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成因子测试汇总文档（回归 + IC）。")
    parser.add_argument("--root", default=None, help=f"结果根目录，默认 {OUTPUT_DIR}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root) if args.root else OUTPUT_DIR
    if not root.exists():
        raise SystemExit(f"结果根目录不存在：{root}")

    names = list_factors(root)
    if not names:
        raise SystemExit(f"在 {root} 下没有找到任何测试结果")

    regressions = [item for item in (load_regression(root, n) for n in names) if item]
    ics = [item for item in (load_ic(root, n) for n in names) if item]
    layered = [item for item in (load_layered(root, n) for n in names) if item]
    if not regressions and not ics and not layered:
        raise SystemExit("没有可汇总的测试结果")

    figure_dir = root / FIGURE_DIR_NAME
    figure_dir.mkdir(parents=True, exist_ok=True)

    if regressions:
        plot_regression_coefficients(regressions, figure_dir / "回归_月度系数_分因子.png")
        plot_regression_cumulative(regressions, figure_dir / "回归_累计溢价_对比.png")
        plot_heatmap(
            regressions, "factor_mean", "Annual mean monthly coefficient (%)",
            figure_dir / "回归_年度系数_热力图.png",
        )
    if ics:
        plot_ic_series(ics, figure_dir / "IC_时间序列_分因子.png")
        plot_ic_cumulative(ics, figure_dir / "IC_累计IC_对比.png")
        plot_heatmap(
            ics, "ic_mean", "Annual mean Rank IC", figure_dir / "IC_年度IC_热力图.png",
        )
    if layered:
        plot_layered_nav(layered, figure_dir / "分层_五组净值_分因子.png")
        plot_long_short_nav(layered, figure_dir / "分层_多空净值_对比.png")
        plot_long_short_heatmap(layered, figure_dir / "分层_年度多空_热力图.png")

    for target in sorted(figure_dir.glob("*.png")):
        print(f"saved: {target}")

    document_path = root / DOC_NAME
    document_path.write_text(
        build_markdown(regressions, ics, layered, names, root), encoding="utf-8"
    )
    print(f"因子数：{len(names)}  ->  {[n.upper() for n in names]}")
    print(f"saved: {document_path}")


if __name__ == "__main__":
    main()
