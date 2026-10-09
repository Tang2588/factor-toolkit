# -*- coding: utf-8 -*-
"""把因子的统计文档编译成 PDF。

    # 编译某个因子的全部三份统计文档（各出一个 PDF）
    python build_report_pdf.py --factor ep

    # 只编译指定的一份
    python build_report_pdf.py --source "output/ep/RankIC/IC统计结果.md"

用法说明：``--factor`` 会依次编译 ``output/<因子名>/`` 下三份统计文档
（RLM统计结果.md、IC统计结果.md、分层回测结果.md），每份各自生成一个 PDF。
三份文档里都只有该因子自己的口径、数据与图表，不含跨因子对比。

需要 xelatex 在 PATH 中（MiKTeX 或 TeX Live 均可）；找不到时只生成 .tex 并提示。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from md_to_latex import convert  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
LATEX_AUX_SUFFIXES = (".aux", ".log", ".out", ".toc", ".synctex.gz")

# 每个因子的三份统计文档：子目录 -> 文档名
FACTOR_DOCS = [
    ("RLM_H1_独立同方差", "RLM统计结果.md"),
    ("RankIC", "IC统计结果.md"),
    ("分层回测", "分层回测结果.md"),
]


def build(source: Path, keep_tex: bool = True) -> dict:
    """把 ``source``（Markdown）编译成同目录同名 PDF。"""
    if not source.exists():
        raise FileNotFoundError("找不到文档：{}".format(source))

    today = _date.today()
    date_text = "{} 年 {} 月 {} 日".format(today.year, today.month, today.day)
    tex_path = source.with_suffix(".tex")
    pdf_path = source.with_suffix(".pdf")

    convert(source, tex_path, None, date_text)

    xelatex = shutil.which("xelatex")
    if xelatex is None:
        print("    未找到 xelatex，跳过 PDF 编译（.tex 已生成，可传到 Overleaf）")
        return {"tex": tex_path, "pdf": None, "compiled": False}

    # 跑两遍：第一遍排版，第二遍生成目录与书签
    for pass_no in (1, 2):
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", tex_path.name],
            cwd=str(source.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        if not pdf_path.exists():
            tail = "\n".join(result.stdout.splitlines()[-12:])
            raise RuntimeError(
                "xelatex 第 {} 遍未能产出 PDF（退出码 {}）。日志尾部：\n{}".format(
                    pass_no, result.returncode, tail
                )
            )

    for suffix in LATEX_AUX_SUFFIXES:
        aux = tex_path.with_suffix(suffix)
        if aux.exists():
            aux.unlink()
    if not keep_tex:
        tex_path.unlink()

    return {"tex": tex_path if keep_tex else None, "pdf": pdf_path, "compiled": True}


def factor_sources(name: str) -> list[Path]:
    """返回某个因子的全部统计文档路径。"""
    factor_dir = OUTPUT_DIR / name
    if not factor_dir.exists():
        raise SystemExit(
            "找不到因子目录：{}\n先运行：python run_all.py --factor {}".format(factor_dir, name)
        )
    found = [factor_dir / sub / doc for sub, doc in FACTOR_DOCS if (factor_dir / sub / doc).exists()]
    if not found:
        raise SystemExit(
            "{} 下没有统计文档。先运行：python run_all.py --factor {}".format(factor_dir, name)
        )
    return found


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把因子的统计文档编译成 PDF。")
    parser.add_argument("--factor", default=None, help="因子名，编译它的全部三份统计文档")
    parser.add_argument("--source", type=Path, default=None, help="指定单份 Markdown 编译")
    parser.add_argument("--drop-tex", action="store_true", help="编译后删掉 .tex，只留 PDF")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.factor and not args.source:
        raise SystemExit("请用 --factor <因子名> 或 --source <Markdown 路径> 指定要编译的文档。")

    if args.source:
        source = args.source if args.source.is_absolute() else (ROOT / args.source)
        sources = [source]
    else:
        sources = factor_sources(args.factor)

    ok = 0
    for source in sources:
        print("编译：{}".format(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source))
        result = build(source, keep_tex=not args.drop_tex)
        if result["pdf"]:
            print("    完成：{} （{:.2f} MB）".format(
                result["pdf"].name, result["pdf"].stat().st_size / 1e6))
            ok += 1
        else:
            print("    PDF 未生成，只有 .tex")
    print("\n共 {} / {} 份编译成功".format(ok, len(sources)))


if __name__ == "__main__":
    main()
