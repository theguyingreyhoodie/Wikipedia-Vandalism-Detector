"""Append-only, date-partitioned parquet tables.

Layout: <base>/date=YYYY-MM-DD/<prefix>_<epoch_ms>.parquet. Replays (`--since`)
can produce duplicates, so readers de-duplicate on a key.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def write_batch(base: Path, prefix: str, rows: list[dict], ts_col: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    stamp = int(time.time() * 1000)
    dates = df[ts_col].map(_date)
    for date, part in df.groupby(dates):
        d = base / f"date={date}"
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / f"{prefix}_{stamp}.parquet", index=False)


def read_table(
    base: Path,
    ts_col: str,
    key: list[str],
    start: float | None = None,
    end: float | None = None,
) -> pd.DataFrame:
    if not base.exists():
        return pd.DataFrame()
    lo = _date(start) if start is not None else None
    hi = _date(end) if end is not None else None
    files = []
    for d in sorted(base.glob("date=*")):
        date = d.name.split("=", 1)[1]
        if (lo and date < lo) or (hi and date > hi):
            continue
        files.extend(sorted(d.glob("*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if start is not None:
        df = df[df[ts_col] >= start]
    if end is not None:
        df = df[df[ts_col] < end]
    return df.drop_duplicates(subset=key, keep="first").sort_values(ts_col).reset_index(drop=True)


def read_edits(cfg, start=None, end=None) -> pd.DataFrame:
    df = read_table(cfg.raw_dir, "timestamp", ["wiki", "revision_new"], start, end)
    if not df.empty:
        df = df[df.wiki == cfg.wiki].reset_index(drop=True)
    return df


def read_reverts(cfg, start=None, end=None) -> pd.DataFrame:
    df = read_table(cfg.reverts_dir, "revert_ts", ["wiki", "rev_id"], start, end)
    if not df.empty:
        df = df[df.wiki == cfg.wiki].reset_index(drop=True)
    return df


def read_predictions(cfg, start=None, end=None) -> pd.DataFrame:
    return read_table(cfg.predictions_dir, "timestamp", ["wiki", "revision_new", "model_version"], start, end)
