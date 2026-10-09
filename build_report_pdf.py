# -*- coding: utf-8 -*-
"""把某个因子的三份统计文档合并成一份，并编译成 PDF。

    # 合并该因子的三份统计文档 → output/<因子名>/因子测试报告.md 与 .pdf
    python build_report_pdf.py --factor ep

    # 只编译指定的单份文档
    python build_report_pdf.py --source "output/ep/RankIC/IC统计结果.md"

合并逻辑：把 ``output/<因子名>/`` 下三份统计文档按「横截面回归 → Rank IC → 分层回测」
的顺序拼成一份，自动把各自的标题降一级（一级标题变成节标题）、把图片的相对路径改成
以因子目录为根。合并出来的文档只包含口径说明、结果表、诊断信息与图，**不含结论文字**。

需要 xelatex 在 PATH 中（MiKTeX 或 TeX Live 均可）；找不到时只生成 .tex 并提示。
"""
from __future__ import annotations

import argparse
import re
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
MERGED_DOC_NAME = "因子测试报告.md"

# 三份统计文档：子目录 -> 文档名
FACTOR_DOCS = [
    ("RLM_H1_独立同方差", "RLM统计结果.md"),
    ("RankIC", "IC统计结果.md"),
    ("分层回测", "分层回测结果.md"),
]

IMAGE_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")


def _rebase_images(text: str, subdir: str) -> str:
    """把 ``./x.png`` 改成 ``<子目录>/x.png``，让合并后的文档能找到图。"""

    def repl(match: re.Match) -> str:
        path = match.group(2).strip()
        if path.startswith(("http://", "https://", "/")):
            return match.group(0)
        if path.startswith("./"):
            path = path[2:]
        return match.group(1) + subdir + "/" + path + match.group(3)

    return IMAGE_RE.sub(repl, text)


def _demote_headings(text: str) -> str:
    """所有标题降一级，让它们成为合并文档里的节与小节。"""
    return re.sub(r"(?m)^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", text)


def merge_factor_docs(factor_dir: Path, name: str) -> tuple[Path, int]:
    """把该因子的三份统计文档合并成一份，返回 (合并文档路径, 合并的份数)。"""
    parts: list[str] = []
    for subdir, doc_name in FACTOR_DOCS:
        source = factor_dir / subdir / doc_name
        if not source.exists():
            continue
        body = _rebase_images(source.read_text(encoding="utf-8"), subdir)
        parts.append(_demote_headings(body).strip())
    if not parts:
        raise SystemExit(
            "{} 下没有统计文档。先运行：python run_all.py --factor {}".format(factor_dir, name)
        )

    today = _date.today()
    header = [
        "# {} 因子测试报告".format(name.upper()),
        "",
        "本报告由 `build_report_pdf.py` 自动合并，数据来自 `run_all.py` 的三步测试结果：",
        "",
        "| 章节 | 来源文档 |",
        "|---|---|",
        "| 横截面回归（Huber RLM） | `RLM_H1_独立同方差/RLM统计结果.md` |",
        "| Rank IC 检验 | `RankIC/IC统计结果.md` |",
        "| 行业内分层回测 | `分层回测/分层回测结果.md` |",
        "",
        "文档中的全部数字、表格与图表均由脚本从测试结果自动生成；报告不含结论文字。",
        "",
        "报告日期：{} 年 {} 月 {} 日".format(today.year, today.month, today.day),
        "",
    ]
    merged_path = factor_dir / MERGED_DOC_NAME
    merged_path.write_text(
        "\n".join(header) + "\n".join(parts) + "\n", encoding="utf-8"
    )
    return merged_path, len(parts)


def build(source: Path, keep_tex: bool = True) -> dict:
    """把 ``source``（Markdown）编译成同目录同名 PDF。"""
    tex_path = source.with_suffix(".tex")
    pdf_path = source.with_suffix(".pdf")
    convert(source, tex_path, None, "")

    xelatex = shutil.which("xelatex")
    if xelatex is None:
        print("    未找到 xelatex，跳过 PDF 编译（.tex 已生成，可传到 Overleaf）")
        return {"tex": tex_path, "pdf": None, "compiled": False}

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把因子的统计文档合并成一份并编译成 PDF。"
    )
    parser.add_argument("--factor", default=None, help="因子名，合并它的三份统计文档后编译")
    parser.add_argument("--source", type=Path, default=None, help="只编译指定的单份 Markdown")
    parser.add_argument("--drop-tex", action="store_true", help="编译后删掉 .tex，只留 PDF")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.factor and not args.source:
        raise SystemExit("请用 --factor <因子名> 或 --source <Markdown 路径>。")

    if args.source:
        source = args.source if args.source.is_absolute() else (ROOT / args.source)
        if not source.exists():
            raise SystemExit("找不到文档：{}".format(source))
        sources = [(source, None)]
    else:
        factor_dir = OUTPUT_DIR / args.factor
        if not factor_dir.exists():
            raise SystemExit(
                "找不到因子目录：{}\n先运行：python run_all.py --factor {}".format(
                    factor_dir, args.factor
                )
            )
        merged, count = merge_factor_docs(factor_dir, args.factor)
        print("已合并 {} 份统计文档：{}".format(count, merged.name))
        sources = [(merged, merged)]

    for source, _ in sources:
        print("编译：{}".format(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source))
        result = build(source, keep_tex=not args.drop_tex)
        if result["pdf"]:
            print("    完成：{} （{:.2f} MB）".format(
                result["pdf"].name, result["pdf"].stat().st_size / 1e6))
        else:
            print("    PDF 未生成，只有 .tex")


if __name__ == "__main__":
    main()