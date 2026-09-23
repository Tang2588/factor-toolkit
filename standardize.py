# -*- coding: utf-8 -*-
"""步骤一：逐日横截面 MAD 去极值与 Z 标准化。

输入一份**原始**因子文件，输出截断后与标准化后两个版本，以及逐日的处理参数。

    python standardize.py --factor D:/某处/roe_raw.parquet
    python standardize.py --factor D:/某处/roe_raw.parquet --name roe

输入格式要求（因子计算层的交付契约）：

    index   = ['date', 'stock_code']
    columns = ['signal']

处理规则（与四个交付版一致）：

    median = 当日横截面中位数
    MAD    = median(|x - median|)
    上下界 = median ± 3 × 1.4826 × MAD
    Z      = (截断后的值 - 当日均值) / 当日标准差

注意：MAD 为 0 时不截断；横截面标准差为 0 时 Z 置为缺失，不填 0。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TOOLKIT_ROOT, factor_output_dir, resolve_factor_path  # noqa: E402
from common.streaming import standardize_factor  # noqa: E402


def run_standardize(factor_input: str, name: str | None = None) -> dict:
    """对一份原始因子做 MAD 去极值与 Z 标准化，返回关键路径与统计量。"""
    raw_path = resolve_factor_path(factor_input)
    factor_name = name or raw_path.stem
    out_dir = factor_output_dir(factor_name) / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)

    mad_path = out_dir / f"{factor_name}_mad.parquet"
    z_path = out_dir / f"{factor_name}.parquet"
    stats_path = out_dir / "processing_stats.parquet"

    print(f"Standardizing: {raw_path}")
    totals = standardize_factor(raw_path, mad_path, z_path, stats_path)
    print(f"  raw valid rows : {totals['raw_valid']:,}")
    print(f"  MAD clipped    : {totals['mad_clipped']:,}")
    print(f"  final valid Z  : {totals['final_valid_z']:,}")
    print(f"  constant cross-sections: {totals['constant_cross_sections']:,}")
    print(f"  saved: {z_path}")
    return {
        "factor_name": factor_name,
        "source_path": raw_path,
        "standardized_path": z_path,
        "mad_path": mad_path,
        "stats_path": stats_path,
        "totals": totals,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐日横截面 MAD 去极值与 Z 标准化。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python standardize.py --factor "D:/某处/roe_raw.parquet"\n'
            '  python standardize.py --factor roe_raw --name roe\n'
        ),
    )
    parser.add_argument("--factor", required=True, help="原始因子文件路径或名称")
    parser.add_argument("--name", default=None, help="因子名，默认取输入文件名")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_standardize(args.factor, args.name)


if __name__ == "__main__":
    main()
