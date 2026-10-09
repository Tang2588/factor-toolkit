# -*- coding: utf-8 -*-
"""因子测试工具箱的共享底层。

本包只提供数据准备函数，不执行业务阶段逻辑，也不写结果文件。

- ``paths``      项目路径与共享常量
- ``parquet_io`` 分片读取 parquet 面板
- ``market``     行情快照清洗、B 股剔除、北交所新旧代码映射
- ``calendar``   月度形成／建仓／退出日历
- ``returns``    前瞻收益
- ``panel``      形成日面板、行业面板与回归面板
"""
from __future__ import annotations

from .calendar import build_monthly_calendar
from .market import (
    apply_security_id,
    b_share_mask,
    derive_bse_mapping,
    extract_code6,
    normalize_market_code,
    prepare_market_snapshot,
)
from .panel import (
    add_neutralized_residual,
    build_regression_panel,
    prepare_formation_panel,
    prepare_industry,
)
from .parquet_io import (
    MarketIndex,
    read_all_dates,
    read_selected_dates,
    scan_market_index,
)
from .paths import (
    FORMAL_TEST_START,
    HUBER_T,
    INDUSTRY_PATH,
    MARKET_PATH,
    MAX_ITER,
    MIN_OBSERVATIONS,
    OUTPUT_DIR,
    STRONG_IC_THRESHOLD,
    TOOLKIT_ROOT,
    TOL,
    factor_output_dir,
    resolve_factor_path,
)
from .returns import build_endpoint_forward_returns

__all__ = [
    "FORMAL_TEST_START",
    "HUBER_T",
    "INDUSTRY_PATH",
    "MARKET_PATH",
    "MAX_ITER",
    "MIN_OBSERVATIONS",
    "OUTPUT_DIR",
    "STRONG_IC_THRESHOLD",
    "TOOLKIT_ROOT",
    "TOL",
    "MarketIndex",
    "add_neutralized_residual",
    "apply_security_id",
    "b_share_mask",
    "build_endpoint_forward_returns",
    "build_monthly_calendar",
    "build_regression_panel",
    "derive_bse_mapping",
    "extract_code6",
    "factor_output_dir",
    "normalize_market_code",
    "prepare_formation_panel",
    "prepare_industry",
    "prepare_market_snapshot",
    "read_all_dates",
    "read_selected_dates",
    "resolve_factor_path",
    "scan_market_index",
]
