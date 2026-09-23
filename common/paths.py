# -*- coding: utf-8 -*-
"""因子测试工具箱的路径与共享常量。

配置解析优先级（从高到低）：

1. 环境变量
2. ``config/paths.local.toml``（本机覆盖，不纳入版本管理）
3. ``config/paths.toml``（仓库内默认值）

工具箱只依赖两样外部输入：基础数据目录、以及到哪里找因子文件。
因子本身与被测因子无关，通过 ``--factor`` 传入名称或路径。
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = TOOLKIT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "paths.toml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "paths.local.toml"

# 找因子文件时的回退目录；实际以配置里的 factor_dirs 为准
FALLBACK_FACTOR_DIRS: list[str] = []
FALLBACK_DATA_ROOT = r"D:\实习生学习项目\基础数据"
FALLBACK_TEST_START = "2020-07-31"

# 回归与检验的共享口径，所有分析脚本必须一致
HUBER_T = 1.345
MAX_ITER = 100
TOL = 1e-8
MIN_OBSERVATIONS = 500
STRONG_IC_THRESHOLD = 0.02


def _read_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config() -> dict:
    """合并仓库默认配置与本机覆盖配置。"""
    config: dict = {}
    config.update(_read_config(DEFAULT_CONFIG_PATH))
    config.update(_read_config(LOCAL_CONFIG_PATH))
    return config


CONFIG = load_config()


def resolve_data_root() -> Path:
    override = os.environ.get("FACTOR_TOOLKIT_DATA_ROOT")
    if override:
        return Path(override)
    return Path(CONFIG.get("data_root", FALLBACK_DATA_ROOT))


DATA_ROOT = resolve_data_root()

MARKET_PATH = DATA_ROOT / "chn_equ_mkt_quotation.parquet"
INDUSTRY_PATH = DATA_ROOT / "chn_equ_indus_sw.parquet"

# 所有分析结果都写到工具箱的 output 目录下，按因子分子目录
OUTPUT_DIR = Path(CONFIG.get("output_dir", str(TOOLKIT_ROOT / "output")))

# 正式测试起点：由覆盖率规则确定，与具体因子无关
FORMAL_TEST_START = str(CONFIG.get("formal_test_start", FALLBACK_TEST_START))


def factor_search_dirs() -> list[Path]:
    """因子文件的搜索目录，来自配置里的 factor_dirs。"""
    dirs = CONFIG.get("factor_dirs", FALLBACK_FACTOR_DIRS)
    return [Path(item) for item in dirs]


def resolve_factor_path(factor: str) -> Path:
    """把因子名或路径解析为因子文件。

    支持三种写法：

    - ``"ep"``                     → 在配置的 factor_dirs 中查找 ``ep.parquet``
    - ``"./某目录/my.parquet"``     → 相对当前工作目录解析
    - ``"D:/其它/my.parquet"``      → 绝对路径直接使用
    """
    candidate = Path(factor)
    if candidate.suffix.lower() == ".parquet" or candidate.exists():
        return candidate.resolve()
    for directory in factor_search_dirs():
        target = directory / f"{factor}.parquet"
        if target.exists():
            return target
    searched = "、".join(str(item) for item in factor_search_dirs()) or "（未配置）"
    raise FileNotFoundError(
        f"找不到因子文件 {factor}.parquet。已搜索：{searched}。"
        f"请用 --factor 传完整路径，或在 config/paths.toml 的 factor_dirs 中补充目录。"
    )


def factor_output_dir(factor_name: str) -> Path:
    """某个因子的结果目录，不存在时创建。"""
    target = OUTPUT_DIR / factor_name
    target.mkdir(parents=True, exist_ok=True)
    return target
