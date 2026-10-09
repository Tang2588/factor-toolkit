# -*- coding: utf-8 -*-
"""把汇总文档《因子测试》编译成 PDF。

    python build_report_pdf.py                 # 默认 output/因子测试.md
    python build_report_pdf.py --source 其它.md

流程：Markdown → LaTeX（md_to_latex）→ xelatex 跑两遍 → 清理中间文件。
需要 xelatex 在 PATH 中（MiKTeX 或 TeX Live 均可）；找不到时只生成 .tex 并提示。

前提：先跑过 ``factor_test_report.py``，让 ``output/因子测试.md`` 与配图存在。
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
DEFAULT_SOURCE = ROOT / "output" / "因子测试.md"
LATEX_AUX_SUFFIXES = (".aux", ".log", ".out", ".toc", ".synctex.gz")


def build(source: Path, keep_tex: bool = True) -> dict:
    """把 ``source``（Markdown）编译成同目录同名 PDF，返回结果路径。"""
    if not source.exists():
        raise FileNotFoundError(
            "找不到汇总文档：{}\n先运行 factor_test_report.py 生成它。".format(source)
        )

    today = _date.today()
    date_text = "{} 年 {} 月 {} 日".format(today.year, today.month, today.day)
    tex_path = source.with_suffix(".tex")
    pdf_path = source.with_suffix(".pdf")

    convert(source, tex_path, None, date_text)
    print("  已生成 LaTeX：{}".format(tex_path.name))

    xelatex = shutil.which("xelatex")
    if xelatex is None:
        print("  未找到 xelatex，跳过 PDF 编译。可把 .tex 传到 Overleaf 编译。")
        return {"tex": tex_path, "pdf": None, "compiled": False}

    # 跑两遍：第一遍排版，第二遍生成目录与书签
    for pass_no in (1, 2):
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", tex_path.name],
            cwd=str(source.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if not pdf_path.exists():
            tail = "\n".join(result.stdout.splitlines()[-12:])
            raise RuntimeError(
                "xelatex 第 {} 遍未能产出 PDF（退出码 {}）。日志尾部：\n{}".format(
                    pass_no, result.returncode, tail
                )
            )
        print("  xelatex 第 {} 遍完成".format(pass_no))

    for suffix in LATEX_AUX_SUFFIXES:
        aux = tex_path.with_suffix(suffix)
        if aux.exists():
            aux.unlink()
    if not keep_tex:
        tex_path.unlink()

    return {"tex": tex_path if keep_tex else None, "pdf": pdf_path, "compiled": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把汇总文档编译成 PDF。")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Markdown 路径，默认 output/因子测试.md")
    parser.add_argument("--drop-tex", action="store_true",
                        help="编译后删掉 .tex，只留 PDF")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source if args.source.is_absolute() else (ROOT / args.source)
    print("编译：{}".format(source))
    result = build(source, keep_tex=not args.drop_tex)
    if result["pdf"]:
        size = result["pdf"].stat().st_size / 1e6
        print("  完成：{} （{:.1f} MB）".format(result["pdf"], size))
    else:
        print("  PDF 未生成，只有 .tex")


if __name__ == "__main__":
    main()