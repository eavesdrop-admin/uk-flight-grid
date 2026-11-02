"""How steady is the busy-day count (oh_day_p90) at the build's sampling rate? Reported, not gated.

The pilot (sample_check.py) compares one month built from every day with the same month at
the production stride. That is a fair test of the mean, but a p90 taken from 15 days is far
noisier than one taken from the build's ~180, so the pilot overstates the p90 error the
report actually carries. This measures it on the build's own days instead:

  * split the build's days into two alternating halves and take each half's p90 per cell.
    Each half has half the days, so its sampling error is about sqrt(2) times the build's;
    the difference of two halves is about 2 times it. ``implied_error = |A - B| / 2`` is
    therefore an estimate of how far the build's p90 sits from a p90 over every day;
  * for named points (the golden points plus WATCH_POINTS) also a bootstrap 5-95% interval
    of the p90 and the daily-count quantiles, so a reader can see why a busy day differs
    from the average (runway alternation shows as a bimodal day count).

The same split is run for the mean, for comparison. Output is ``p90_stability.json`` next to
the grid; it holds real per-point numbers, so it stays in the private bucket with the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from h3.api import basic_int as h3i

from grid_schema import CELL_RES

TOLERANCE = 0.10
ABS_TOLERANCE = 1.0  # flights a day, as in sample_check.py
TOP_N = 1_000
BOOTSTRAP_REPS = 1_000
BOOTSTRAP_SEED = 20261102

# Places worth watching that are not golden points (no expectation is asserted for them).
WATCH_POINTS = [
    {"name": "Windsor SL4, EGLL 09L final approach", "lat": 51.47534, "lon": -0.62761},
    {"name": "Rhayader LD6, mid-Wales partial coverage", "lat": 52.31050, "lon": -3.50896},
]


def _day_matrix(cells: np.ndarray, day_frames: Sequence[pl.LazyFrame], col: str = "oh_n") -> np.ndarray:
    """(cells x days) daily counts; a cell absent from a day counts as zero."""
    order = np.argsort(cells)
    sorted_cells = cells[order]
    mat = np.zeros((len(cells), len(day_frames)), dtype=np.float64)
    for i, lf in enumerate(day_frames):
        day = lf.select("cell", col).filter(pl.col("cell").is_in(sorted_cells.tolist())).collect()
        if day.height == 0:
            continue
        idx = np.searchsorted(sorted_cells, day["cell"].to_numpy())
        mat[order[idx], i] = day[col].to_numpy()
    return mat


def _agrees(error: np.ndarray, full: np.ndarray) -> np.ndarray:
    return (error <= TOLERANCE * full) | (error <= ABS_TOLERANCE)


def _split(mat: np.ndarray, stat) -> dict[str, np.ndarray]:
    full = stat(mat)
    a, b = stat(mat[:, 0::2]), stat(mat[:, 1::2])
    return {"full": full, "a": a, "b": b, "implied_error": np.abs(a - b) / 2}


def _p90(m: np.ndarray) -> np.ndarray:
    return np.percentile(m, 90, axis=1)


def _mean(m: np.ndarray) -> np.ndarray:
    return m.mean(axis=1)


def _summary(s: dict[str, np.ndarray]) -> dict[str, Any]:
    full, err = s["full"], s["implied_error"]
    rel = np.divide(err, full, out=np.zeros_like(err), where=full > 0)
    return {
        "within_fraction": round(float(_agrees(err, full).mean()), 4) if len(full) else None,
        "median_rel_error": round(float(np.median(rel)), 4) if len(rel) else None,
        "p90_rel_error": round(float(np.quantile(rel, 0.9)), 4) if len(rel) else None,
        "max_rel_error": round(float(rel.max()), 4) if len(rel) else None,
    }


def stability(cells: pl.DataFrame, day_frames: Sequence[pl.LazyFrame], points: Sequence[dict]) -> dict[str, Any]:
    """The report written to p90_stability.json. ``cells`` is the built cells table."""
    if len(day_frames) < 2:
        return {"days": len(day_frames), "note": "Needs at least two days to split; nothing measured."}
    top = cells.sort("oh_day_mean", descending=True).head(TOP_N)
    top_cells = top["h3_r9"].cast(pl.UInt64).to_numpy()
    named = []
    for p in points:
        c = h3i.latlng_to_cell(p["lat"], p["lon"], CELL_RES)
        named.append((p["name"], c, cells.filter(pl.col("h3_r9") == np.int64(np.uint64(c))).height > 0))
    named_cells = np.array([c for _, c, _ in named], dtype=np.uint64)

    all_cells = np.unique(np.concatenate([top_cells, named_cells]))
    mat = _day_matrix(all_cells, day_frames)
    pos = {int(c): i for i, c in enumerate(all_cells)}
    top_rows = mat[[pos[int(c)] for c in top_cells]]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    detail = []
    for name, c, in_grid in named:
        row = mat[pos[int(c)]][None, :]
        p, m = _split(row, _p90), _split(row, _mean)
        n = row.shape[1]
        boot = np.percentile(row[0][rng.integers(0, n, size=(BOOTSTRAP_REPS, n))], 90, axis=1)
        detail.append({
            "name": name,
            "in_grid": in_grid,
            "p90": round(float(p["full"][0]), 1),
            "p90_halves": [round(float(p["a"][0]), 1), round(float(p["b"][0]), 1)],
            "p90_implied_error": round(float(p["implied_error"][0]), 1),
            "p90_bootstrap_5_95": [round(float(np.quantile(boot, 0.05)), 1), round(float(np.quantile(boot, 0.95)), 1)],
            "p90_within": bool(_agrees(p["implied_error"], p["full"])[0]),
            "mean": round(float(m["full"][0]), 2),
            "mean_implied_error": round(float(m["implied_error"][0]), 2),
            "days_active": int((row[0] > 0).sum()),
            "day_quantiles": {q: round(float(np.quantile(row[0], float(q))), 1) for q in ("0.1", "0.25", "0.5", "0.75", "0.9")},
        })

    return {
        "days": int(mat.shape[1]),
        "tolerance": TOLERANCE,
        "abs_tolerance": ABS_TOLERANCE,
        "top_n": int(len(top_cells)),
        "top_p90": _summary(_split(top_rows, _p90)),
        "top_mean": _summary(_split(top_rows, _mean)),
        "points": detail,
        "note": "Reported, not gated. implied_error = |half A - half B| / 2 estimates the build's sampling error.",
    }


def write(out_dir: Path, cells: pl.DataFrame, day_frames: Sequence[pl.LazyFrame], points: Sequence[dict]) -> dict[str, Any]:
    report = stability(cells, day_frames, points)
    (Path(out_dir) / "p90_stability.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
