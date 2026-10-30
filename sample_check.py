"""Sampled-vs-full pilot check (plan section 8, check 6; decision 13).

    uv run python sample_check.py --month 2026-09 --extract-dir out/extract --work out/pilot

Builds one full calendar month (every day, stride 1) and the same month at the production
stride (every 4th day by ordinal, the same days ``plan_days.py sampled`` picks), then compares
per-cell ``oh_day_mean``:

  * every golden point that has at least MIN_GOLDEN_MEAN legs a day in the full build must
    agree within TOLERANCE (10%), and at least one must qualify;
  * for the TOP_N (1,000) busiest cells of the full build, at least TOP_MIN_FRACTION (90%)
    must agree within TOLERANCE. A cell absent from the sampled grid counts as zero.

p90 and ``oh_days_active`` are reported for the golden points, not gated (sampling noise is
expected there). The result is ``sampling_check.json``; validate_grid.py reads it and
refuses a stride > 1 build without a passing one made with the same builder version and
stride. If it fails, raise the sampling rate (stride 2) or build every day.

The thresholds are this script's reading of "must agree within 10%": the plan does not say
what share of the top 1,000 cells has to, so that share is a parameter and is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from h3.api import basic_int as h3i

import build_grid
from grid_schema import CELL_RES
from plan_days import month_days, sampled_in
from reference import DATA_DIR

TOLERANCE = 0.10
TOP_N = 1_000
TOP_MIN_FRACTION = 0.90
MIN_GOLDEN_MEAN = 1.0
MIN_FULL_MONTH_SHARE = 0.90  # of the calendar month's days must be usable, or it is not a full month
DEFAULT_STRIDE = 4


def rel_error(full: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    return np.abs(sampled - full) / full


def compare(
    full: pl.DataFrame,
    sampled: pl.DataFrame,
    golden: list[dict],
    *,
    tolerance: float = TOLERANCE,
    top_n: int = TOP_N,
    top_min_fraction: float = TOP_MIN_FRACTION,
    min_golden_mean: float = MIN_GOLDEN_MEAN,
) -> dict[str, Any]:
    """Compare two cells tables (columns h3_r9, oh_day_mean, oh_day_p90, oh_days_active)."""
    s_mean = dict(zip(sampled["h3_r9"].to_list(), sampled["oh_day_mean"].to_list(), strict=True))

    top = full.sort("oh_day_mean", descending=True).head(top_n)
    f = top["oh_day_mean"].to_numpy().astype(float)
    s = np.array([s_mean.get(c, 0.0) for c in top["h3_r9"].to_list()], dtype=float)
    err = rel_error(f, s) if len(f) else np.array([])
    top_within = float((err <= tolerance).mean()) if len(err) else 0.0

    by_cell = {r["h3_r9"]: r for r in full.iter_rows(named=True)}
    s_rows = {r["h3_r9"]: r for r in sampled.iter_rows(named=True)}
    golden_out, golden_fail = [], []
    for p in golden:
        cell = h3i.latlng_to_cell(p["lat"], p["lon"], CELL_RES)
        fr = by_cell.get(cell)
        sr = s_rows.get(cell)
        row: dict[str, Any] = {"name": p["name"], "full_oh_day_mean": fr and round(float(fr["oh_day_mean"]), 3),
                               "sampled_oh_day_mean": round(float(sr["oh_day_mean"]), 3) if sr else 0.0}
        if fr is None or fr["oh_day_mean"] < min_golden_mean:
            row["gated"] = False
        else:
            e = abs(row["sampled_oh_day_mean"] - fr["oh_day_mean"]) / fr["oh_day_mean"]
            row.update(gated=True, rel_error=round(float(e), 4), within=bool(e <= tolerance),
                       full_oh_day_p90=int(fr["oh_day_p90"]), sampled_oh_day_p90=int(sr["oh_day_p90"]) if sr else 0,
                       full_days_active=int(fr["oh_days_active"]), sampled_days_active=int(sr["oh_days_active"]) if sr else 0)
            if not row["within"]:
                golden_fail.append(p["name"])
        golden_out.append(row)

    n_gated = sum(1 for g in golden_out if g["gated"])
    reasons = []
    if n_gated == 0:
        reasons.append(f"no golden point had {min_golden_mean} legs a day in the full build")
    if golden_fail:
        reasons.append(f"golden points outside {tolerance:.0%}: {', '.join(golden_fail)}")
    if len(f) == 0:
        reasons.append("full build has no cells")
    elif top_within < top_min_fraction:
        reasons.append(f"only {top_within:.1%} of the {len(f)} busiest cells within {tolerance:.0%} (need {top_min_fraction:.0%})")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "golden": golden_out,
        "golden_gated": n_gated,
        "top_cells": len(f),
        "top_within_fraction": round(top_within, 4),
        "top_median_error": round(float(np.median(err)), 4) if len(err) else None,
        "top_p90_error": round(float(np.quantile(err, 0.9)), 4) if len(err) else None,
        "top_max_error": round(float(err.max()), 4) if len(err) else None,
    }


def run(
    month: str,
    extract_dir: Path,
    work: Path,
    cache: Path,
    ref_dir: Path,
    golden_path: Path,
    stride: int = DEFAULT_STRIDE,
    workers: int = 4,
    log=print,
) -> dict[str, Any]:
    year, mon = (int(x) for x in month.split("-"))
    wanted = [d.isoformat() for d in month_days(year, mon)]
    have = {p.stem for p in Path(extract_dir).rglob("*.parquet")}
    present = [d for d in wanted if d in have]

    full_dir = build_grid.build(extract_dir, work / "full", cache, ref_dir, days=present, day_stride=1, workers=workers, log=log)
    full_meta = json.loads((full_dir / "meta.json").read_text())
    used = full_meta["input_days"]
    sub_days = sampled_in(used, stride)

    reasons = []
    need = int(len(wanted) * MIN_FULL_MONTH_SHARE)
    if len(used) < need:
        reasons.append(f"only {len(used)} usable days of {len(wanted)} in {month} (need {need}); not a full month")
    if len(sub_days) < 2:
        reasons.append(f"only {len(sub_days)} sampled days at stride {stride}")

    result: dict[str, Any] = {
        "month": month, "stride": stride, "builder_version": build_grid.BUILDER_VERSION,
        "tolerance": TOLERANCE, "top_n": TOP_N, "top_min_fraction": TOP_MIN_FRACTION,
        "days_full": len(used), "days_sampled": len(sub_days), "sampled_days": sub_days,
        "month_days": len(wanted), "checked_at": date.today().isoformat(),
    }
    if reasons:
        return {**result, "passed": False, "reasons": reasons}

    sub_dir = build_grid.build(extract_dir, work / "sampled", cache, ref_dir, days=sub_days, day_stride=stride, workers=workers, log=log)
    cols = ["h3_r9", "oh_day_mean", "oh_day_p90", "oh_days_active"]
    cmp = compare(
        pl.read_parquet(full_dir / "cells.parquet", columns=cols),
        pl.read_parquet(sub_dir / "cells.parquet", columns=cols),
        json.loads(Path(golden_path).read_text())["points"],
    )
    return {**result, **cmp}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--month", required=True, help="YYYY-MM, a complete month already extracted")
    ap.add_argument("--extract-dir", type=Path, default=Path("out/extract"))
    ap.add_argument("--work", type=Path, default=Path("out/pilot"), help="holds full/ and sampled/ builds")
    ap.add_argument("--cache", type=Path, default=Path("out/agg"))
    ap.add_argument("--reference-dir", type=Path, default=DATA_DIR / "raw" / "ourairports")
    ap.add_argument("--golden", type=Path, default=Path(__file__).parent / "golden_points.json")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, help="default: <work>/sampling_check.json")
    a = ap.parse_args(argv)

    result = run(a.month, a.extract_dir, a.work, a.cache, a.reference_dir, a.golden, a.stride, a.workers)
    out = a.out or a.work / "sampling_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {a.month}, {result['days_sampled']} sampled of {result['days_full']} days -> {out}")
    for r in result.get("reasons", []):
        print(f"  {r}")
    # Golden points and summary errors only (no grid cells), so safe for the public log.
    for k in ("top_within_fraction", "top_median_error", "top_p90_error", "top_max_error"):
        if k in result:
            print(f"  {k}: {result[k]}")
    for g in result.get("golden", []):
        print(f"  golden {g['name']}: full {g['full_oh_day_mean']} sampled {g['sampled_oh_day_mean']}"
              + (f" err {g['rel_error']:.1%} p90 {g['full_oh_day_p90']}/{g['sampled_oh_day_p90']}"
                 f" active {g['full_days_active']}/{g['sampled_days_active']}" if g.get("gated") else " (not gated)"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
