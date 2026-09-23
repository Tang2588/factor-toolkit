# -*- coding: utf-8 -*-
"""把所有已测试因子的关键指标汇总成一张横向对比表。

    python summarize_factors.py

扫描 ``output/<因子名>/`` 下的各步骤结果，抽取核心指标输出 markdown 与 csv。
缺失的步骤留空，不会报错，因此可以边测边看。

对比表只放「能跨因子比大小的数」；极值月份、条件数、降权比例这类诊断量留在
各因子自己的结果目录里，不塞进汇总表。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUTPUT_DIR  # noqa: E402

RLM_DIR = "RLM_H1_独立同方差"
IC_DIR = "RankIC"
LAYERED_DIR = "分层回测"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def collect_one(factor_dir: Path) -> dict | None:
    """读取单个因子的各步骤结果，返回一行指标。"""
    name = factor_dir.name
    row: dict = {"因子": name}

    rlm = _read(factor_dir / RLM_DIR / "rlm_overall_summary.csv")
    if rlm is not None and len(rlm):
        item = rlm.iloc[0]
        row["回归月份"] = int(item.get("months", 0)) or np.nan
        row["回归月均系数"] = _finite(item.get("factor_mean"))
        row["回归t值"] = _finite(item.get("factor_iid_t"))
        row["回归p值"] = _finite(item.get("factor_iid_p"))
        row["回归正系数比例"] = _finite(item.get("factor_positive_ratio"))

    ic = _read(factor_dir / IC_DIR / "ic_overall_summary.csv")
    if ic is not None and len(ic):
        item = ic.iloc[0]
        row["IC均值"] = _finite(item.get("ic_mean"))
        row["IC标准差"] = _finite(item.get("ic_std"))
        row["IR"] = _finite(item.get("ir"))
        row["IC_t值"] = _finite(item.get("ic_t"))
        row["IC_p值"] = _finite(item.get("ic_p"))
        row["IC正比例"] = _finite(item.get("ic_positive_ratio"))

    layered = _read(factor_dir / LAYERED_DIR / "layered_performance.csv")
    if layered is not None and len(layered) >= 2:
        long_short = layered.iloc[-1]
        top_group = layered.iloc[-2]
        row["多空年化收益"] = _finite(long_short.get("annual_return"))
        row["多空夏普"] = _finite(long_short.get("sharpe"))
        row["多空最大回撤"] = _finite(long_short.get("max_drawdown"))
        row["多空胜率"] = _finite(long_short.get("win_ratio"))
        row["最高组年化收益"] = _finite(top_group.get("annual_return"))
        row["最高组信息比率"] = _finite(top_group.get("info_ratio"))
        row["平均换手率"] = _finite(long_short.get("avg_turnover"))

    if len(row) == 1:
        return None
    return row


def consistency(row: pd.Series) -> str:
    """三个检验的方向是否一致：回归系数、IC 均值、多空收益。"""
    keys = ["回归月均系数", "IC均值", "多空年化收益"]
    signs = [np.sign(row.get(key, np.nan)) for key in keys]
    signs = [s for s in signs if np.isfinite(s) and s != 0]
    if not signs:
        return "数据不足"
    if len(signs) < len(keys):
        return f"部分完成（{len(signs)}/3）"
    return "一致" if len(set(signs)) == 1 else "不一致"


def build_table(root: Path) -> pd.DataFrame:
    rows = []
    if root.exists():
        for factor_dir in sorted(item for item in root.iterdir() if item.is_dir()):
            row = collect_one(factor_dir)
            if row:
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["因子", "一致性"])
    table = pd.DataFrame(rows)
    table["一致性"] = table.apply(consistency, axis=1)
    return table


def _fmt(value, kind: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if kind == "pct":
        return f"{number * 100:+.2f}%"
    if kind == "num":
        return f"{number:+.3f}"
    return f"{number:,.0f}"


COLUMNS = [
    ("回归月均系数", "pct"),
    ("回归t值", "num"),
    ("回归p值", "num"),
    ("IC均值", "num"),
    ("IC标准差", "num"),
    ("IR", "num"),
    ("IC_t值", "num"),
    ("IC_p值", "num"),
    ("多空年化收益", "pct"),
    ("多空夏普", "num"),
    ("多空最大回撤", "pct"),
    ("平均换手率", "pct"),
]


def to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "# 因子对比表\n\n还没有任何因子的测试结果。\n"

    header = "| 因子 | " + " | ".join(name for name, _ in COLUMNS) + " | 一致性 |"
    divider = "|---|" + "---:|" * len(COLUMNS) + "---|"
    lines = [
        "# 因子对比表",
        "",
        "由 `summarize_factors.py` 自动生成。缺失的步骤显示为 -，说明该因子还没跑到那一步。",
        "",
        header,
        divider,
    ]
    for _, row in table.iterrows():
        cells = [_fmt(row.get(name), kind) for name, kind in COLUMNS]
        lines.append(f"| {row['因子']} | " + " | ".join(cells) + f" | {row['一致性']} |")

    lines += [
        "",
        "说明：",
        "",
        "- **回归月均系数**：因子每高一个横截面标准差，下一持有期收益的平均变化。因子已按横截面标准化，故可跨因子比较；",
        "- **IR**：IC 均值 ÷ IC 标准差（月频，未年化）。显著性可用 `t = IR × √月数` 判断；",
        "- **多空年化收益**：行业内分层回测的 Q5 − Q1，市值加权，不含交易成本；",
        "- **一致性**：回归系数、IC 均值、多空收益三者方向是否一致。不一致说明「排序信息」与「线性幅度证据」冲突，需要人工判断，不宜直接使用；",
        "- 极值月份、条件数、降权比例等诊断量保留在各因子自己的结果目录中，不进对比表。",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总所有已测因子的关键指标。")
    parser.add_argument("--root", default=None, help=f"结果根目录，默认 {OUTPUT_DIR}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.root) if args.root else OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    table = build_table(root)

    markdown_path = root / "因子对比表.md"
    csv_path = root / "因子对比表.csv"
    markdown_path.write_text(to_markdown(table), encoding="utf-8")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"因子数：{len(table)}")
    print(f"saved: {markdown_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
