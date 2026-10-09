# -*- coding: utf-8 -*-
"""因子标准测试流程：一键跑完横截面回归、Rank IC 与行业内分层回测。

    python run_all.py --factor ep

    # 只跑其中几步
    python run_all.py --factor ep --only ic,layered

结果按因子分目录写到 ``output/<因子名>/`` 下：

    RLM_H1_独立同方差/  横截面回归
    RankIC/        Rank IC 检验
    分层回测/       行业内分层回测

输入契约：因子文件必须**已经做过**逐日横截面 MAD 去极值与 Z 标准化，
格式为 ``index = [date, stock_code]``、``columns = [signal]``。
标准化属于因子计算层（各因子仓库的第 5 步），本工具箱不重复实现——
这样同一套去极值口径全项目只有一份代码。

三个步骤使用完全相同的样本、前瞻收益与控制变量，结果可以直接横向对照。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cross_section_rlm  # noqa: E402
import layered_backtest  # noqa: E402
import rank_ic  # noqa: E402
from common import OUTPUT_DIR, resolve_factor_path  # noqa: E402

ALL_STEPS = ["rlm", "ic", "layered"]
STEP_LABELS = {
    "rlm": "横截面回归（Huber RLM）",
    "ic": "Rank IC 检验",
    "layered": "行业内分层回测",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="因子标准测试流程：标准化、横截面回归、Rank IC、分层回测。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python run_all.py --factor ep\n'
            '  python run_all.py --factor ep --only ic,layered\n'
        ),
    )
    parser.add_argument("--factor", required=True, help="因子文件路径，或配置中的因子名")
    parser.add_argument("--name", default=None, help="因子名，默认取因子文件名")
    parser.add_argument(
        "--only",
        default=",".join(ALL_STEPS),
        help=f"只跑指定步骤，逗号分隔，可选：{','.join(ALL_STEPS)}（默认全部）",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help="单月最小观测数，默认使用共享常量 500",
    )
    return parser.parse_args(argv)


def run_pipeline(
    factor_input: str,
    name: str | None = None,
    steps: list[str] | None = None,
    min_observations: int | None = None,
) -> dict:
    """按顺序执行标准测试流程，返回各步骤的关键路径。"""
    steps = steps or list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise ValueError(f"未知步骤：{unknown}，可选 {ALL_STEPS}")

    result: dict = {"steps": steps}

    factor_path = resolve_factor_path(factor_input)
    factor_name = name or factor_path.stem

    result["factor_name"] = factor_name
    result["factor_path"] = factor_path
    print(f"\n因子：{factor_name}  ({factor_path})")
    print(f"结果目录：{OUTPUT_DIR / factor_name}\n")

    common_args = ["--factor", str(factor_path), "--out-root", str(OUTPUT_DIR)]
    if min_observations is not None:
        common_args += ["--min-observations", str(min_observations)]

    failures: list[str] = []
    for index, step in enumerate(steps, start=1):
        print("=" * 70)
        print(f"步骤 {index}/{len(steps)}  {STEP_LABELS[step]}")
        print("=" * 70)
        try:
            if step == "rlm":
                cross_section_rlm.main(common_args)
            elif step == "ic":
                rank_ic.main(common_args)
            elif step == "layered":
                layered_backtest.main(common_args)
            result[step] = "ok"
        except Exception as error:  # noqa: BLE001
            result[step] = f"failed: {error}"
            failures.append(step)
            print(f"！！步骤 {step} 失败：{error}")
            traceback.print_exc()
        print()

    result["failures"] = failures
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    steps = [item.strip() for item in args.only.split(",") if item.strip()]
    summary = run_pipeline(
        args.factor,
        name=args.name,
        steps=steps,
        min_observations=args.min_observations,
    )
    print("=" * 70)
    print("完成")
    print("=" * 70)
    for step in steps:
        print(f"  {STEP_LABELS[step]:<24} {summary.get(step)}")
    if summary["failures"]:
        raise SystemExit(f"以下步骤失败：{summary['failures']}")


if __name__ == "__main__":
    main()
