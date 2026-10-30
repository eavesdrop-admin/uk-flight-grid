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


P90_CHUNK = 200_000
FOLD_DAYS = 8


def _p90_matrix(cells_arr: np.ndarray, day_vals: Sequence[pl.LazyFrame], col: str) -> np.ndarray:
    """90th percentile over ALL days per kept cell, zeros included (numpy linear interpolation).

    Filled one day at a time into a (cells x days) matrix, so only that matrix and one day's
    column are ever in memory. A cell absent from a day counts as zero for that day.
    """
    n, d = len(cells_arr), len(day_vals)
    mat = np.zeros((n, d), dtype=np.uint16)
    for i, lf in enumerate(day_vals):
        day = lf.select("cell", col).collect()
        cell = day["cell"].to_numpy()
        vals = day[col].to_numpy()
        idx = np.searchsorted(cells_arr, cell)
        kept = idx < n
        kept[kept] = cells_arr[idx[kept]] == cell[kept]
        v = vals[kept]
        if len(v) and v.max() > np.iinfo(np.uint16).max:
            raise ValueError(f"{col} exceeds {np.iinfo(np.uint16).max} legs in one cell on one day")
        mat[idx[kept], i] = v
    out = np.zeros(n, dtype=np.float64)
    for lo in range(0, n, P90_CHUNK):
        out[lo : lo + P90_CHUNK] = np.percentile(mat[lo : lo + P90_CHUNK], 90, axis=1)
    return out


def _fold_sum(days: Sequence[pl.LazyFrame], kept: pl.LazyFrame, keys: list[str]) -> pl.DataFrame:
    """Sum column n by keys across days, for kept cells only, folding FOLD_DAYS days at a time."""
    acc: pl.DataFrame | None = None
    for lo in range(0, len(days), FOLD_DAYS):
        batch = [lf.join(kept, on="cell", how="semi").select(*keys, "n") for lf in days[lo : lo + FOLD_DAYS]]
        if acc is not None:
            batch.insert(0, acc.lazy())
        acc = pl.concat(batch).group_by(keys).agg(pl.col("n").sum()).collect()
    assert acc is not None
    return acc


def merge_days(days: Sequence[DayAgg], min_mean: float = MIN_MEAN_PER_DAY) -> pl.DataFrame:
    """Merge in-memory per-day aggregates. See merge_frames."""
    return merge_frames(
        [a.cells.lazy() for a in days], [a.alt_hist.lazy() for a in days], [a.attrib.lazy() for a in days], min_mean
    )


def merge_frames(
    cells: Sequence[pl.LazyFrame],
    alt_hist: Sequence[pl.LazyFrame],
    attrib: Sequence[pl.LazyFrame],
    min_mean: float = MIN_MEAN_PER_DAY,
) -> pl.DataFrame:
    """Merge per-day aggregates into the final cells table (CELLS_SCHEMA).

    Takes one lazy frame per day for each DayAgg table (e.g. scans of the day cache), and
    streams the cross-day group-bys, so memory does not grow with every day's rows at once.
    Means divide by len(cells) (every observed day, including days with no flight in the cell).
    Only cells with nb_day_mean >= min_mean or oh_day_mean >= min_mean are kept.
    """
    d = len(cells)
    if d == 0:
        raise ValueError("no days to merge")
    value_cols = [c for c in cells[0].collect_schema().names() if c != "cell"]
    # Running totals, folded a few days at a time: a single group-by over every day's rows
    # holds them all at once (about 80 MB a day), which a 90-day build cannot afford.
    sums = [pl.col(c).sum() for c in (*value_cols, "oh_days_active")]
    tot: pl.DataFrame | None = None
    for lo in range(0, d, FOLD_DAYS):
        batch = [
            lf.select(
                "cell",
                *[pl.col(c).cast(pl.Float64) for c in value_cols],
                (pl.col("oh_n") > 0).cast(pl.UInt32).alias("oh_days_active"),
            )
            for lf in cells[lo : lo + FOLD_DAYS]
        ]
        if tot is not None:
            batch.insert(0, tot.lazy())
        tot = pl.concat(batch).group_by("cell").agg(sums).collect()
    assert tot is not None
    tot = tot.filter((pl.col("nb_n") / d >= min_mean - 1e-12) | (pl.col("oh_n") / d >= min_mean - 1e-12)).sort("cell")
    cells_arr = tot["cell"].to_numpy()
    kept = tot.select("cell").lazy()

    # p90 over all days, zeros included
    p90 = {prefix: _p90_matrix(cells_arr, cells, f"{prefix}_n") for prefix in ("oh", "nb")}

    # median altitude of overhead legs
    hist = (
        _fold_sum(alt_hist, kept, ["cell", "bin"])
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
        _fold_sum(attrib, kept, ["cell", "airport", "runway", "op"])
        .join(tot.select("cell", "oh_n"), on="cell", how="inner")
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
