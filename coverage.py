"""Receiver coverage: where is the ADS-B archive actually able to see low aircraft?

Empty hexes must never be read as "no flights". Coverage is judged per res-7 cell (about
5 km across) from the observed points of the cell and its k=1 ring:

  floor_ft     5th percentile of observed altitude (AMSL, 100 ft resolution, rounded up).
               A receiver that only hears aircraft above 4,000 ft says nothing about
               what flies at 1,000 ft.
  continuity   observed aircraft-time / (observed + time lost in gaps over 120 s that span
               the cell). Low values mean tracks drop out in the cell.
  sample_points  observed points in the ring.

Status (see classify()):
  good                  floor <= 1,500 ft, enough samples, and continuity >= 0.85
                        (or fewer than 50 leg crossings, too few to judge continuity)
  partial_low_altitude  enough samples, and floor 1,500 to 4,000 ft, or floor <= 1,500 ft with continuity 0.60 to 0.85
  insufficient          everything else. A cell with no row at all is also insufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl
from h3.api import basic_int as h3i

from grid_schema import (
    COVERAGE_RES,
    COVERAGE_SCHEMA,
    MAX_COUNTED_ALT_FT,
    STATUS_GOOD,
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    conform,
)

GOOD_FLOOR_FT = 1_500
PARTIAL_FLOOR_FT = 4_000
GOOD_CONTINUITY = 0.85
PARTIAL_CONTINUITY = 0.60
FEW_CROSSINGS = 50
MIN_SAMPLE_POINTS = 500
FLOOR_PERCENTILE = 0.05
GAP_S = 120.0  # a silence longer than this is a gap
LEG_BREAK_S = 15 * 60.0  # same as segment.LEG_GAP_S: longer silences end the leg
GAP_STEP_M = 1_000.0  # spacing of the points that spread a gap over the cells it spans
HIST_BIN_FT = 100
HIST_FLOOR_FT = -1_000
HIST_NBINS = 120  # -1,000 .. 11,000 ft (Stage A keeps up to 10,000)
M_PER_DEG = 111_195.0


@dataclass
class CoverageDay:
    stats: pl.DataFrame  # cell, n_points, crossings, obs_s, gap_s
    hist: pl.DataFrame  # cell, bin, n


def _cell_array(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (h3i.latlng_to_cell(a, b, COVERAGE_RES) for a, b in zip(lat.tolist(), lon.tolist())),
        dtype=np.uint64,
        count=len(lat),
    )


def coverage_day(points: pl.DataFrame) -> CoverageDay:
    """Coverage statistics for one Stage A day (columns uid, leg_flag, t_utc, lat, lon, alt_ft)."""
    p = points.sort("uid", "t_utc")
    t = p["t_utc"].dt.epoch("ms").to_numpy() / 1000.0
    lat = p["lat"].to_numpy()
    lon = p["lon"].to_numpy()
    alt = p["alt_ft"].to_numpy().astype(np.float64)
    uid = p["uid"].to_numpy()
    leg_flag = p["leg_flag"].to_numpy()
    n = len(p)
    cell = _cell_array(lat, lon)

    dt = np.diff(t, prepend=t[0])
    new_leg = np.r_[True, uid[1:] != uid[:-1]] | leg_flag | (dt > LEG_BREAK_S)
    leg_id = np.cumsum(new_leg)

    # consecutive pairs inside one leg: pair i joins point i-1 -> i
    pair = ~new_leg
    pair[0] = False
    i = np.flatnonzero(pair)
    a, b = i - 1, i
    pdt = t[b] - t[a]

    observed = pdt <= GAP_S
    obs_cell = cell[a[observed]]
    obs_s = pdt[observed]

    # gaps spanning cells: spread the lost time along the straight line, only when both ends are low
    gap = (~observed) & (alt[a] <= MAX_COUNTED_ALT_FT) & (alt[b] <= MAX_COUNTED_ALT_FT)
    ga, gb = a[gap], b[gap]
    gap_cells: list[np.ndarray] = []
    gap_secs: list[np.ndarray] = []
    if len(ga):
        dy = (lat[gb] - lat[ga]) * M_PER_DEG
        dx = (lon[gb] - lon[ga]) * M_PER_DEG * np.cos(np.radians(lat[ga]))
        steps = np.maximum(1, np.ceil(np.hypot(dx, dy) / GAP_STEP_M)).astype(np.int64)
        rep = np.repeat(np.arange(len(ga)), steps)
        k = np.arange(len(rep)) - np.repeat(np.cumsum(steps) - steps, steps)
        f = (k + 0.5) / steps[rep]
        glat = lat[ga][rep] + f * (lat[gb] - lat[ga])[rep]
        glon = lon[ga][rep] + f * (lon[gb] - lon[ga])[rep]
        gap_cells.append(_cell_array(glat, glon))
        gap_secs.append((pdt[gap] / steps)[rep])

    frames = [
        pl.DataFrame({"cell": cell}).group_by("cell").agg(pl.len().cast(pl.UInt32).alias("n_points")),
        pl.DataFrame({"cell": cell, "leg": leg_id})
        .unique()
        .group_by("cell")
        .agg(pl.len().cast(pl.UInt32).alias("crossings")),
        pl.DataFrame({"cell": obs_cell, "s": obs_s}).group_by("cell").agg(pl.col("s").sum().alias("obs_s")),
    ]
    if gap_cells:
        frames.append(
            pl.DataFrame({"cell": gap_cells[0], "s": gap_secs[0]}).group_by("cell").agg(pl.col("s").sum().alias("gap_s"))
        )
    stats = frames[0]
    for fr in frames[1:]:
        stats = stats.join(fr, on="cell", how="full", coalesce=True)  # gap time can fall in cells with no points
    if "gap_s" not in stats.columns:
        stats = stats.with_columns(pl.lit(0.0).alias("gap_s"))
    stats = stats.with_columns(
        pl.col("n_points").fill_null(0), pl.col("crossings").fill_null(0),
        pl.col("obs_s").fill_null(0.0), pl.col("gap_s").fill_null(0.0),
    )

    b_ = np.clip(np.floor((alt - HIST_FLOOR_FT) / HIST_BIN_FT), 0, HIST_NBINS - 1).astype(np.uint8)
    hist = pl.DataFrame({"cell": cell, "bin": b_}).group_by("cell", "bin").agg(pl.len().cast(pl.UInt32).alias("n"))
    return CoverageDay(stats.sort("cell"), hist.sort("cell", "bin"))


def classify(
    floor_ft: float,
    continuity: float,
    sample_points: int,
    crossings: int,
    min_sample_points: int = MIN_SAMPLE_POINTS,
) -> str:
    """Status for one cell. Few samples is always insufficient, whatever the floor."""
    if sample_points < min_sample_points:
        return STATUS_INSUFFICIENT
    continuity_ok = continuity >= GOOD_CONTINUITY or crossings < FEW_CROSSINGS
    if floor_ft <= GOOD_FLOOR_FT and continuity_ok:
        return STATUS_GOOD
    if GOOD_FLOOR_FT < floor_ft <= PARTIAL_FLOOR_FT:
        return STATUS_PARTIAL  # counts are a lower bound: low aircraft may be unheard
    if floor_ft <= GOOD_FLOOR_FT and continuity >= PARTIAL_CONTINUITY:
        return STATUS_PARTIAL  # low aircraft heard, but tracks drop out
    return STATUS_INSUFFICIENT


def finalise(days: Sequence[CoverageDay], min_sample_points: int = MIN_SAMPLE_POINTS) -> pl.DataFrame:
    """Merge days, pool each cell with its k=1 ring, classify. Returns COVERAGE_SCHEMA.

    Rows exist only for cells that themselves contain at least one observed point. Gap time
    that falls in a cell with no points still counts against the ring cells around it.
    """
    stats = (
        pl.concat([d.stats for d in days])
        .group_by("cell")
        .agg(pl.col("n_points").sum(), pl.col("crossings").sum(), pl.col("obs_s").sum(), pl.col("gap_s").sum())
    )
    hist = pl.concat([d.hist for d in days]).group_by("cell", "bin").agg(pl.col("n").sum())
    cells = stats.filter(pl.col("n_points") > 0)["cell"].to_list()
    ring = pl.DataFrame(
        [(c, nb) for c in cells for nb in h3i.grid_disk(c, 1)],
        schema={"cell": pl.UInt64, "nbr": pl.UInt64},
        orient="row",
    )
    ring_stats = (
        ring.join(stats.rename({"cell": "nbr"}), on="nbr", how="inner")
        .group_by("cell")
        .agg(
            pl.col("n_points").sum().alias("sample_points"),
            pl.col("crossings").sum().alias("crossings"),
            pl.col("obs_s").sum().alias("obs_s"),
            pl.col("gap_s").sum().alias("gap_s"),
        )
    )
    ring_hist = (
        ring.join(hist.rename({"cell": "nbr"}), on="nbr", how="inner")
        .group_by("cell", "bin")
        .agg(pl.col("n").sum())
        .sort("cell", "bin")
        .with_columns(pl.col("n").cum_sum().over("cell").alias("cum"), pl.col("n").sum().over("cell").alias("total"))
        .filter(pl.col("cum") >= pl.col("total") * FLOOR_PERCENTILE)
        .group_by("cell")
        .agg(pl.col("bin").first())
        # upper edge of the bin: the conservative reading of a 100 ft bin
        .with_columns(((pl.col("bin").cast(pl.Int32) + 1) * HIST_BIN_FT + HIST_FLOOR_FT).alias("floor_ft"))
        .select("cell", "floor_ft")
    )
    own = pl.DataFrame({"cell": cells}, schema={"cell": pl.UInt64})
    df = (
        own.join(ring_stats, on="cell", how="left")
        .join(ring_hist, on="cell", how="left")
        .with_columns(
            pl.when(pl.col("obs_s") + pl.col("gap_s") > 0)
            .then(pl.col("obs_s") / (pl.col("obs_s") + pl.col("gap_s")))
            .otherwise(1.0)
            .alias("continuity")
        )
    )
    status = [
        classify(f, c, int(n), int(x), min_sample_points)
        for f, c, n, x in zip(df["floor_ft"], df["continuity"], df["sample_points"], df["crossings"])
    ]
    df = df.with_columns(
        pl.Series("status", status),
        pl.col("cell").cast(pl.Int64).alias("h3_r7"),
    )
    return conform(df, COVERAGE_SCHEMA).sort("h3_r7")


def downgrade(coverage: pl.DataFrame, cells_r7: Sequence[int]) -> pl.DataFrame:
    """Move the given res-7 cells down one step (good -> partial -> insufficient).

    Used when the airport that dominates a cell is under-captured by the receivers.
    """
    targets = pl.Series("t", list(cells_r7), dtype=pl.Int64)
    step = (
        pl.when(pl.col("status") == STATUS_GOOD).then(pl.lit(STATUS_PARTIAL))
        .when(pl.col("status") == STATUS_PARTIAL).then(pl.lit(STATUS_INSUFFICIENT))
        .otherwise(pl.col("status"))
    )
    return coverage.with_columns(
        pl.when(pl.col("h3_r7").is_in(targets.implode())).then(step).otherwise(pl.col("status")).alias("status")
    )
