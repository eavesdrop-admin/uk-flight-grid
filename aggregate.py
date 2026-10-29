"""Per-day aggregation of leg-cell rows, and the merge across days.

A "day" here is one Stage A file (a UTC day). Per cell and day we count legs by altitude
band, by night, and (overhead only) by inferred final approach, plus a 100 ft altitude
histogram and counts per (airport, runway, op). Merging across days takes means over ALL
observed days, so a cell that sees a flight on two days out of ninety has a low mean and a
p90 of zero. A day with no flights counts, because it was observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from grid_schema import (
    BAND_EDGES_FT,
    CELLS_SCHEMA,
    LOCAL_TZ,
    MIN_MEAN_PER_DAY,
    NIGHT_END_HOUR,
    NIGHT_START_HOUR,
    conform,
)

ALT_BIN_FT = 100
ALT_FLOOR_FT = -1_000  # bin 0 starts here (AMSL; deep-low barometric readings can go below 0)
ALT_NBINS = 80  # -1,000 .. 7,000 ft

BAND_COLS = ("lt1000", "1000_3000", "3000_7000")


# ----------------------------------------------------------------- night


def local_hour(t_epoch_s: np.ndarray | pl.Series) -> np.ndarray:
    """Hour of day (0-23) on the Europe/London wall clock for UTC epoch seconds."""
    s = t_epoch_s if isinstance(t_epoch_s, pl.Series) else pl.Series(np.asarray(t_epoch_s, dtype=np.float64))
    ms = (s.cast(pl.Float64) * 1000).round().cast(pl.Int64)
    local = ms.cast(pl.Datetime("ms", "UTC")).dt.convert_time_zone(LOCAL_TZ)
    return local.dt.hour().to_numpy()


def night_mask(t_epoch_s: np.ndarray | pl.Series) -> np.ndarray:
    """True for 23:00 to 06:59:59 local time. Correct across the March and October clock changes."""
    h = local_hour(t_epoch_s)
    return (h >= NIGHT_START_HOUR) | (h < NIGHT_END_HOUR)


def band_index(alt_ft: np.ndarray) -> np.ndarray:
    """0: below 1,000 ft, 1: 1,000 to 3,000, 2: 3,000 to 7,000 (callers drop >= 7,000)."""
    return np.digitize(alt_ft, BAND_EDGES_FT[:2]).astype(np.int8)


def alt_bin(alt_ft: np.ndarray) -> np.ndarray:
    b = np.floor((np.asarray(alt_ft, dtype=np.float64) - ALT_FLOOR_FT) / ALT_BIN_FT)
    return np.clip(b, 0, ALT_NBINS - 1).astype(np.uint8)


# ------------------------------------------------------------------- day


@dataclass
class DayAgg:
    cells: pl.DataFrame  # cell, oh_*, nb_* counts (UInt32)
    alt_hist: pl.DataFrame  # cell, bin, n   (overhead legs)
    attrib: pl.DataFrame  # cell, airport, runway, op, n   (overhead legs)


def _counts(df: pl.DataFrame, prefix: str, with_inferred: bool) -> pl.DataFrame:
    exprs = [
        pl.len().cast(pl.UInt32).alias(f"{prefix}_n"),
        *[(pl.col("band") == i).sum().cast(pl.UInt32).alias(f"{prefix}_b{i}") for i in range(3)],
        pl.col("night").sum().cast(pl.UInt32).alias(f"{prefix}_night"),
    ]
    if with_inferred:
        exprs.append(pl.col("inferred").sum().cast(pl.UInt32).alias(f"{prefix}_inf"))
    return df.group_by("cell").agg(exprs)


def _prep(rows: pl.DataFrame) -> pl.DataFrame:
    alt = rows["alt_ft"].to_numpy()
    return rows.with_columns(
        pl.Series("band", band_index(alt), dtype=pl.Int8),
        pl.Series("night", night_mask(rows["t"]), dtype=pl.Boolean),
    )


def aggregate_day(overhead: pl.DataFrame, nearby: pl.DataFrame, legs: pl.DataFrame) -> DayAgg:
    """Aggregate one day.

    overhead: leg_id, cell (UInt64), alt_ft, t (epoch s), inferred   - one row per leg per cell
    nearby:   leg_id, cell, alt_ft, t                                - same, for the 2 km set
    legs:     leg_id, airport, runway, op                            - attribution of each leg
    """
    oh = _prep(overhead)
    nb = _prep(nearby)

    nb_c = _counts(nb, "nb", False)
    oh_c = _counts(oh, "oh", True)
    cells = nb_c.join(oh_c, on="cell", how="full", coalesce=True)
    for col in cells.columns:
        if col != "cell":
            cells = cells.with_columns(pl.col(col).fill_null(0).cast(pl.UInt32))

    hist = (
        oh.with_columns(pl.Series("bin", alt_bin(oh["alt_ft"].to_numpy()), dtype=pl.UInt8))
        .group_by("cell", "bin")
        .agg(pl.len().cast(pl.UInt32).alias("n"))
    )
    attrib = (
        oh.join(legs.select("leg_id", "airport", "runway", "op"), on="leg_id", how="left")
        .with_columns(pl.col("airport").fill_null(""), pl.col("runway").fill_null(""))
        .group_by("cell", "airport", "runway", "op")
        .agg(pl.len().cast(pl.UInt32).alias("n"))
    )
    return DayAgg(cells.sort("cell"), hist.sort("cell", "bin"), attrib.sort("cell", "airport", "runway", "op"))


# ----------------------------------------------------------------- merge


def _p90_matrix(day_idx: np.ndarray, cell_idx: np.ndarray, vals: np.ndarray, n_cells: int, n_days: int) -> np.ndarray:
    """90th percentile over ALL n_days per cell, zeros included (numpy linear interpolation)."""
    out = np.zeros(n_cells, dtype=np.float64)
    chunk = 200_000
    for lo in range(0, n_cells, chunk):
        hi = min(lo + chunk, n_cells)
        sel = (cell_idx >= lo) & (cell_idx < hi)
        mat = np.zeros((hi - lo, n_days), dtype=np.int32)
        mat[cell_idx[sel] - lo, day_idx[sel]] = vals[sel]
        out[lo:hi] = np.percentile(mat, 90, axis=1)
    return out


def merge_days(days: Sequence[DayAgg], min_mean: float = MIN_MEAN_PER_DAY) -> pl.DataFrame:
    """Merge per-day aggregates into the final cells table (CELLS_SCHEMA).

    Means divide by len(days) (every observed day, including days with no flight in the cell).
    Only cells with nb_day_mean >= min_mean or oh_day_mean >= min_mean are kept.
    """
    d = len(days)
    if d == 0:
        raise ValueError("no days to merge")
    stacked = pl.concat([a.cells.with_columns(pl.lit(i, dtype=pl.UInt16).alias("day")) for i, a in enumerate(days)])
    value_cols = [c for c in stacked.columns if c not in ("cell", "day")]
    tot = stacked.group_by("cell").agg(
        *[pl.col(c).sum().cast(pl.Float64).alias(c) for c in value_cols],
        (pl.col("oh_n") > 0).sum().alias("oh_days_active"),
    )
    tot = tot.filter(
        (pl.col("nb_n") / d >= min_mean - 1e-12) | (pl.col("oh_n") / d >= min_mean - 1e-12)
    ).sort("cell")
    cells_arr = tot["cell"].to_numpy()
    n = len(cells_arr)

    # p90 over all days, zeros included
    p90 = {}
    for prefix in ("oh", "nb"):
        s = stacked.join(tot.select("cell"), on="cell", how="semi")
        cell_idx = np.searchsorted(cells_arr, s["cell"].to_numpy())
        p90[prefix] = _p90_matrix(
            s["day"].to_numpy().astype(np.int64), cell_idx, s[f"{prefix}_n"].to_numpy().astype(np.int64), n, d
        )

    # median altitude of overhead legs
    hist = (
        pl.concat([a.alt_hist for a in days])
        .join(tot.select("cell"), on="cell", how="semi")
        .group_by("cell", "bin")
        .agg(pl.col("n").sum())
        .sort("cell", "bin")
        .with_columns(
            pl.col("n").cum_sum().over("cell").alias("cum"),
            pl.col("n").sum().over("cell").alias("total"),
        )
        .filter(pl.col("cum") * 2 >= pl.col("total"))
        .group_by("cell")
        .agg(pl.col("bin").first())
        .with_columns(
            (pl.col("bin").cast(pl.Int32) * ALT_BIN_FT + ALT_FLOOR_FT + ALT_BIN_FT // 2).alias("oh_alt_p50_ft")
        )
        .select("cell", "oh_alt_p50_ft")
    )

    # top-2 (airport, runway, op) among overhead legs
    att = (
        pl.concat([a.attrib for a in days])
        .join(tot.select("cell", "oh_n"), on="cell", how="inner")
        .group_by("cell", "airport", "runway", "op", "oh_n")
        .agg(pl.col("n").sum())
        .sort(["cell", "n", "airport", "runway", "op"], descending=[False, True, False, False, False])
        .with_columns((pl.int_range(pl.len()).over("cell") + 1).alias("rank"))
        .filter(pl.col("rank") <= 2)
        .with_columns((pl.col("n") * 100.0 / pl.col("oh_n")).round().alias("share"))
    )

    def top(rank: int, p: str) -> pl.DataFrame:
        return att.filter(pl.col("rank") == rank).select(
            "cell",
            pl.when(pl.col("airport") == "").then(None).otherwise(pl.col("airport")).alias(f"{p}_airport"),
            pl.when(pl.col("runway") == "").then(None).otherwise(pl.col("runway")).alias(f"{p}_runway"),
            pl.col("op").alias(f"{p}_op"),
            pl.col("share").alias(f"{p}_share"),
        )

    out = (
        tot.with_columns(
            pl.Series("oh_day_p90", np.rint(p90["oh"])),
            pl.Series("nb_day_p90", np.rint(p90["nb"])),
        )
        .join(hist, on="cell", how="left")
        .join(top(1, "top1"), on="cell", how="left")
        .join(top(2, "top2"), on="cell", how="left")
    )
    out = out.with_columns(
        (pl.col("cell").cast(pl.UInt64).cast(pl.Int64)).alias("h3_r9"),
        (pl.col("oh_n") / d).alias("oh_day_mean"),
        (pl.col("oh_b0") / d).alias("oh_lt1000"),
        (pl.col("oh_b1") / d).alias("oh_1000_3000"),
        (pl.col("oh_b2") / d).alias("oh_3000_7000"),
        (pl.col("oh_night") / d).alias("oh_night_mean"),
        (pl.col("oh_inf") / pl.col("oh_n")).fill_nan(None).fill_null(0.0).alias("oh_inferred_share"),
        (pl.col("nb_n") / d).alias("nb_day_mean"),
        (pl.col("nb_b0") / d).alias("nb_lt1000"),
        (pl.col("nb_b1") / d).alias("nb_1000_3000"),
        (pl.col("nb_b2") / d).alias("nb_3000_7000"),
        (pl.col("nb_night") / d).alias("nb_night_mean"),
    )
    return conform(out, CELLS_SCHEMA).sort("h3_r9")
