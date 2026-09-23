# -*- coding: utf-8 -*-
"""逐日横截面 MAD 去极值与 Z 标准化。

因子文件可能很大（数千万行），因此按 parquet 批次读取、按交易日处理，
避免一次性载入内存。
"""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MAD_SCALE = 1.4826
CLIP_MULTIPLIER = 3.0


def _open_indexed_writer(path: Path, frame: pd.DataFrame) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=True)
    return pq.ParquetWriter(path, table.schema)


def _write_indexed(writer: pq.ParquetWriter, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, schema=writer.schema, preserve_index=True)
    writer.write_table(table)


def _standardize_one_day(day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """单日横截面处理：先截断到 median ± 3×1.4826×MAD，再做 Z 标准化。"""
    values = pd.to_numeric(day["signal"], errors="coerce")
    valid = values.notna()
    mad_signal = pd.Series(np.nan, index=day.index, dtype="float64")
    z_signal = pd.Series(np.nan, index=day.index, dtype="float64")
    median = mad = lower = upper = mean = std = np.nan
    clipped_count = 0

    if valid.any():
        valid_values = values.loc[valid]
        median = float(valid_values.median())
        mad = float((valid_values - median).abs().median())
        scale = mad * MAD_SCALE
        if scale > 0:
            lower = median - CLIP_MULTIPLIER * scale
            upper = median + CLIP_MULTIPLIER * scale
            clipped = valid_values.clip(lower=lower, upper=upper)
        else:
            # MAD 为 0 时不截断
            lower, upper = -np.inf, np.inf
            clipped = valid_values
        clipped_count = int((clipped != valid_values).sum())
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        mad_signal.loc[valid] = clipped
        if std > 0:
            z_signal.loc[valid] = (clipped - mean) / std

    columns = ["date", "stock_code", "signal"]
    mad_out = (
        pd.DataFrame({"date": day["date"], "stock_code": day["stock_code"], "signal": mad_signal})
        .loc[:, columns]
        .set_index(["date", "stock_code"])
    )
    z_out = (
        pd.DataFrame({"date": day["date"], "stock_code": day["stock_code"], "signal": z_signal})
        .loc[:, columns]
        .set_index(["date", "stock_code"])
    )
    row = {
        "date": day["date"].iloc[0],
        "n_valid": int(valid.sum()),
        "median": median,
        "mad": mad,
        "lower": lower,
        "upper": upper,
        "n_clipped": clipped_count,
        "mean_after_mad": mean,
        "std_after_mad": std,
    }
    return mad_out, z_out, row


def standardize_factor(
    raw_path: Path,
    mad_path: Path,
    z_path: Path,
    stats_path: Path,
    final_path: Path | None = None,
    batch_size: int = 50_000,
) -> dict[str, int]:
    """逐日 MAD 去极值 + Z 标准化。

    输入 ``raw_path`` 需为 ``index = [date, stock_code]``、``columns = [signal]``。
    输出 ``mad_path``（截断后）与 ``z_path``（标准化后）两个同格式文件，
    并把每日处理参数写入 ``stats_path``。
    """
    pf = pq.ParquetFile(raw_path)
    mad_writer = None
    z_writer = None
    carry = pd.DataFrame()
    stats_rows: list[dict] = []
    totals = {
        "raw_valid": 0,
        "mad_clipped": 0,
        "final_valid_z": 0,
        "constant_cross_sections": 0,
    }

    def flush(process: pd.DataFrame) -> None:
        nonlocal mad_writer, z_writer
        for _, day in process.groupby("date", sort=False):
            mad_out, z_out, row = _standardize_one_day(day)
            stats_rows.append(row)
            totals["raw_valid"] += row["n_valid"]
            totals["mad_clipped"] += row["n_clipped"]
            totals["final_valid_z"] += int(z_out["signal"].notna().sum())
            totals["constant_cross_sections"] += int(
                row["n_valid"] > 0 and not (row["std_after_mad"] > 0)
            )
            if mad_writer is None:
                mad_writer = _open_indexed_writer(mad_path, mad_out)
            if z_writer is None:
                z_writer = _open_indexed_writer(z_path, z_out)
            _write_indexed(mad_writer, mad_out)
            _write_indexed(z_writer, z_out)

    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            chunk = batch.to_pandas(ignore_metadata=True)
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["stock_code"] = chunk["stock_code"].astype("string")
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            # 最后一天可能还没读全，留到下一批
            last_date = chunk["date"].iloc[-1]
            ready = chunk["date"].ne(last_date)
            flush(chunk.loc[ready])
            carry = chunk.loc[~ready].copy()
        if not carry.empty:
            flush(carry)
    finally:
        if mad_writer is not None:
            mad_writer.close()
        if z_writer is not None:
            z_writer.close()

    pd.DataFrame(stats_rows).sort_values("date").to_parquet(stats_path, index=False)
    if final_path is not None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(z_path, final_path)
    return totals
