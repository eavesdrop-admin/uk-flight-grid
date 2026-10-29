"""Validate a built grid against the plan's section 8 checks. Exit 1 if any gate fails.

    uv run python validate_grid.py out/grid/2026-09-30 \
        --sampling-check out/pilot/sampling_check.json --extract-dir out/extract

Writes ``validation.json`` next to the grid (``--out`` to move it). The file always gets
written, pass or fail; a failing build must not be published to the loader's ``grid/`` prefix.

Gates (a gate is a failure unless it is listed as skipped or deferred):

  days_gate        days_observed meets the build gate for the day stride (80 at stride 4,
                   300 at stride 1; the loader enforces the same numbers).
  golden:<name>    each point in golden_points.json: airport attribution, runway, traffic
                   level, median altitude, days active, receiver coverage.
  capture_ratio    observed / DfT movements within 0.90 to 1.35 for the 15 busiest England
                   and Wales airports in the DfT file.
  coverage         no 'good' cell with floor_ft above 1,500; statuses valid; table not empty.
  baro_share       under 20% of counted points use barometric altitude (needs --extract-dir).
  sampling_check   stride > 1 needs a passing sampled-vs-full pilot check made with the same
                   builder version and stride (see sample_check.py).

Skipped when its input is absent, and recorded as such: baro_share, sampling_check at stride 1.
Deferred (need data this repo does not hold): the Spearman check against DEFRA Lden and the
prod-versus-staging replica check. They are listed in validation.json under ``deferred``.

The report holds real per-point numbers, so keep it private: ``--summary-only`` prints just
check names and pass/fail, which is what the public CI log gets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from h3.api import basic_int as h3i

from grid_schema import COVERAGE_RES, COVERAGE_STATUSES, MAX_COUNTED_ALT_FT, STATUS_GOOD, STATUS_INSUFFICIENT, CELL_RES

VALIDATOR_VERSION = 1
HERE = Path(__file__).parent
DEFAULT_GOLDEN = HERE / "golden_points.json"

CAPTURE_TOP_N = 15
CAPTURE_LO, CAPTURE_HI = 0.90, 1.35
GOOD_FLOOR_FT_MAX = 1_500
BARO_SHARE_MAX = 0.20
# Same numbers as the Eavesdrop loader (apps/workers/scripts/ingest_adsb_lol_flight_grid.py).
GATE_DAYS_STRIDE_1 = 300
GATE_DAYS_STRIDE_4 = 80

# ICAO prefixes that are not England or Wales: Scotland (EGP, EGQ), Northern Ireland (EGA),
# Channel Islands / Isle of Man (EGJ, EGNS) are not in the DfT file anyway.
NON_EW_PREFIXES = ("EGP", "EGQ", "EGA")

DEFERRED = [
    "spearman_vs_defra_lden: needs the DEFRA aircraft-noise mirror, which lives in the Eavesdrop database",
    "prod_vs_staging_replica: needs the same day extracted from the staging release; run by hand once per builder version",
]


@dataclass
class Check:
    name: str
    status: str  # pass | fail | skipped
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def required_days(day_stride: int) -> int:
    """Same rule as the loader: 300 at stride 1, 80 at stride 4, else max(80, ceil(300 / stride))."""
    if day_stride <= 1:
        return GATE_DAYS_STRIDE_1
    if day_stride == 4:
        return GATE_DAYS_STRIDE_4
    return max(GATE_DAYS_STRIDE_4, math.ceil(GATE_DAYS_STRIDE_1 / day_stride))


# ------------------------------------------------------------------ checks


def check_days_gate(meta: dict) -> Check:
    need = required_days(int(meta["day_stride"]))
    have = int(meta["days_observed"])
    ok = have >= need
    return Check(
        "days_gate",
        "pass" if ok else "fail",
        f"{have} days observed, gate {need} at stride {meta['day_stride']}",
        {"days_observed": have, "required": need},
    )


def is_england_wales(icao: str) -> bool:
    return icao.startswith("EG") and not icao.startswith(NON_EW_PREFIXES)


def check_capture_ratio(airports: pl.DataFrame, top_n: int = CAPTURE_TOP_N, lo: float = CAPTURE_LO, hi: float = CAPTURE_HI) -> Check:
    """Observed movements against DfT for the busiest England and Wales airports."""
    ew = airports.filter(pl.col("icao").map_elements(is_england_wales, return_dtype=pl.Boolean) & pl.col("dft_atm").is_not_null())
    top = ew.sort("dft_atm", descending=True).head(top_n)
    if top.height < top_n:
        return Check("capture_ratio", "fail", f"only {top.height} England and Wales airports have a DfT figure, need {top_n}")
    rows, bad = [], []
    for r in top.iter_rows(named=True):
        ratio = r["capture_ratio"]
        rows.append({"icao": r["icao"], "capture_ratio": None if ratio is None else round(float(ratio), 3), "dft_atm": r["dft_atm"]})
        if ratio is None or not (lo <= ratio <= hi):
            bad.append(f"{r['icao']} {'n/a' if ratio is None else f'{ratio:.2f}'}")
    return Check(
        "capture_ratio",
        "fail" if bad else "pass",
        f"{len(bad)} of {top_n} outside {lo:.2f} to {hi:.2f}: {', '.join(bad)}" if bad else f"all {top_n} within {lo:.2f} to {hi:.2f}",
        {"airports": rows},
    )


def check_coverage(coverage: pl.DataFrame) -> Check:
    if coverage.height == 0:
        return Check("coverage", "fail", "coverage table is empty")
    problems = []
    unknown = sorted(set(coverage["status"].to_list()) - set(COVERAGE_STATUSES))
    if unknown:
        problems.append(f"unknown statuses {unknown}")
    bad_good = coverage.filter((pl.col("status") == STATUS_GOOD) & (pl.col("floor_ft") > GOOD_FLOOR_FT_MAX)).height
    if bad_good:
        problems.append(f"{bad_good} 'good' cells with floor_ft above {GOOD_FLOOR_FT_MAX}")
    n_good = coverage.filter(pl.col("status") == STATUS_GOOD).height
    if n_good == 0:
        problems.append("no 'good' cells at all")
    counts = {s: int((coverage["status"] == s).sum()) for s in COVERAGE_STATUSES}
    insufficient_share = counts[STATUS_INSUFFICIENT] / coverage.height
    return Check(
        "coverage",
        "fail" if problems else "pass",
        "; ".join(problems) if problems else f"{coverage.height} res-{COVERAGE_RES} cells, {insufficient_share:.0%} insufficient (reported, not gated)",
        {"status_counts": counts, "insufficient_share": round(insufficient_share, 4)},
    )


def check_baro_share(extract_dir: Path, days: list[str]) -> Check:
    """Share of counted points (below 7,000 ft) whose altitude is barometric rather than geometric."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = {p.stem: p for p in Path(extract_dir).rglob("*.parquet")}
    missing = [d for d in days if d not in files]
    if missing:
        return Check("baro_share", "fail", f"extract files missing for {len(missing)} days (first: {missing[0]})")
    total = baro = 0
    for d in days:
        t = pq.read_table(files[d], columns=["alt_ft", "alt_src"])
        t = t.filter(pc.less(t["alt_ft"], MAX_COUNTED_ALT_FT))
        total += t.num_rows
        baro += pc.sum(pc.equal(t["alt_src"], "baro").cast("int64")).as_py() or 0
    if total == 0:
        return Check("baro_share", "fail", "no counted points in the extract files")
    share = baro / total
    return Check(
        "baro_share",
        "pass" if share < BARO_SHARE_MAX else "fail",
        f"{share:.1%} barometric (limit {BARO_SHARE_MAX:.0%})",
        {"baro_share": round(share, 4), "points": total},
    )


def check_sampling(meta: dict, sampling: dict | None) -> Check:
    stride = int(meta["day_stride"])
    if stride <= 1:
        return Check("sampling_check", "skipped", "full build (stride 1): nothing sampled")
    if sampling is None:
        return Check("sampling_check", "fail", "no sampled-vs-full pilot check supplied (run sample_check.py on the pilot month)")
    problems = []
    if sampling.get("passed") is not True:
        problems.append("the pilot check itself did not pass")
    if sampling.get("builder_version") != meta["builder_version"]:
        problems.append(f"pilot used builder v{sampling.get('builder_version')}, this build is v{meta['builder_version']}")
    if sampling.get("stride") != stride:
        problems.append(f"pilot used stride {sampling.get('stride')}, this build is stride {stride}")
    return Check(
        "sampling_check",
        "fail" if problems else "pass",
        "; ".join(problems) if problems else f"pilot month {sampling.get('month')} agreed within {sampling.get('tolerance'):.0%}",
        {"month": sampling.get("month"), "builder_version": sampling.get("builder_version")},
    )


# ------------------------------------------------------------ golden points


def lookup_point(cells: pl.LazyFrame | pl.DataFrame, coverage: pl.DataFrame, lat: float, lon: float) -> dict[str, Any]:
    c9 = h3i.latlng_to_cell(lat, lon, CELL_RES)
    c7 = h3i.cell_to_parent(c9, COVERAGE_RES)
    lf = cells.lazy() if isinstance(cells, pl.DataFrame) else cells
    rows = lf.filter(pl.col("h3_r9") == c9).collect()
    cov = coverage.filter(pl.col("h3_r7") == c7)
    return {
        "h3_r9": c9,
        "cell": rows.row(0, named=True) if rows.height else None,
        # A res-7 cell with no row has no observed point: insufficient.
        "coverage_status": cov["status"][0] if cov.height else STATUS_INSUFFICIENT,
        "floor_ft": int(cov["floor_ft"][0]) if cov.height else None,
    }


def _in(x: float | None, rng: list[float]) -> bool:
    return x is not None and rng[0] <= x <= rng[1]


def check_golden_point(point: dict, found: dict[str, Any], days_observed: int) -> Check:
    exp = point["expect"]
    cell, status, floor = found["cell"], found["coverage_status"], found["floor_ft"]
    fails: list[str] = []

    if cell is None and not exp.get("cell_may_be_absent"):
        fails.append("no cell row (fewer than 0.1 legs a day)")

    if cell is not None:
        if "top1_airport" in exp and cell["top1_airport"] != exp["top1_airport"]:
            fails.append(f"top airport {cell['top1_airport']}, expected {exp['top1_airport']}")
        for key in ("top1_op", "top1_runway"):
            if key in exp:
                want = exp[key] if isinstance(exp[key], list) else [exp[key]]
                if cell[key] not in want:
                    fails.append(f"{key.replace('top1_', 'top ')} {cell[key]}, expected {'/'.join(want)}")
        if exp.get("no_airport_attribution") and cell["top1_airport"] is not None:
            fails.append(f"attributed to {cell['top1_airport']}, expected no airport")
        if "oh_day_mean_max" in exp and cell["oh_day_mean"] > exp["oh_day_mean_max"]:
            fails.append(f"{cell['oh_day_mean']:.1f} legs a day, expected at most {exp['oh_day_mean_max']}")
        if "oh_alt_p50_ft" in exp and not _in(cell["oh_alt_p50_ft"], exp["oh_alt_p50_ft"]):
            fails.append(f"median altitude {cell['oh_alt_p50_ft']} ft, expected {exp['oh_alt_p50_ft'][0]} to {exp['oh_alt_p50_ft'][1]}")
        if "days_active_frac" in exp:
            frac = cell["oh_days_active"] / days_observed if days_observed else 0.0
            if not _in(frac, exp["days_active_frac"]):
                fails.append(f"active {frac:.2f} of days, expected {exp['days_active_frac'][0]} to {exp['days_active_frac'][1]}")
    # A "minimum" is a claim of traffic, so an absent cell fails it (a missing row is below 0.1 a day).
    if "oh_day_mean_min" in exp:
        mean = cell["oh_day_mean"] if cell is not None else 0.0
        if mean < exp["oh_day_mean_min"]:
            fails.append(f"{mean:.1f} legs a day, expected at least {exp['oh_day_mean_min']}")

    if "coverage_in" in exp and status not in exp["coverage_in"]:
        fails.append(f"coverage {status}, expected {'/'.join(exp['coverage_in'])}")
    if "not_good_unless_floor_le" in exp and status == STATUS_GOOD and (floor is None or floor > exp["not_good_unless_floor_le"]):
        fails.append(f"'good' coverage with floor {floor} ft")

    observed = {"coverage_status": status, "floor_ft": floor, "h3_r9": found["h3_r9"]}
    if cell is not None:
        observed.update({k: cell[k] for k in ("oh_day_mean", "oh_alt_p50_ft", "oh_days_active", "top1_airport", "top1_runway", "top1_op", "top1_share")})
    return Check(f"golden:{point['name']}", "fail" if fails else "pass", "; ".join(fails), {"observed": observed})


def check_golden(cells: pl.LazyFrame, coverage: pl.DataFrame, golden: dict, days_observed: int) -> list[Check]:
    return [
        check_golden_point(p, lookup_point(cells, coverage, p["lat"], p["lon"]), days_observed)
        for p in golden["points"]
    ]


# --------------------------------------------------------------------- run


def validate(grid_dir: Path, golden_path: Path, sampling: dict | None, extract_dir: Path | None) -> dict:
    grid_dir = Path(grid_dir)
    meta = json.loads((grid_dir / "meta.json").read_text())
    cells = pl.scan_parquet(grid_dir / "cells.parquet")
    coverage = pl.read_parquet(grid_dir / "coverage.parquet")
    airports = pl.read_parquet(grid_dir / "airports.parquet")
    golden = json.loads(Path(golden_path).read_text())

    checks = [check_days_gate(meta)]
    checks += check_golden(cells, coverage, golden, int(meta["days_observed"]))
    checks += [check_capture_ratio(airports), check_coverage(coverage)]
    if extract_dir is not None:
        checks.append(check_baro_share(extract_dir, meta["input_days"]))
    else:
        checks.append(Check("baro_share", "skipped", "no --extract-dir given"))
    sampling_check = check_sampling(meta, sampling)
    checks.append(sampling_check)

    failures = [c.name for c in checks if c.status == "fail"]
    passed = not failures
    cov = next(c for c in checks if c.name == "coverage")
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        # Keys the Eavesdrop loader reads: `passed`, and `sampling_check_passed` when stride > 1.
        "sampling_check_passed": sampling_check.status in ("pass", "skipped"),
        "validator_version": VALIDATOR_VERSION,
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_id": meta["build_id"],
        "window_start": meta["window_start"],
        "window_end": meta["window_end"],
        "days_observed": meta["days_observed"],
        "day_stride": meta["day_stride"],
        "builder_version": meta["builder_version"],
        "insufficient_share": cov.data.get("insufficient_share"),
        "failures": failures,
        "skipped": [c.name for c in checks if c.status == "skipped"],
        "deferred": DEFERRED,
        "checks": [asdict(c) for c in checks],
        "sampling_check_detail": sampling,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("grid_dir", type=Path)
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--sampling-check", type=Path, help="sampling_check.json from sample_check.py")
    ap.add_argument("--extract-dir", type=Path, help="Stage A directory, for the barometric-share check")
    ap.add_argument("--out", type=Path, help="where to write validation.json (default: next to the grid)")
    ap.add_argument("--summary-only", action="store_true", help="print check names and pass/fail only (public CI logs)")
    a = ap.parse_args(argv)

    sampling = json.loads(a.sampling_check.read_text()) if a.sampling_check and a.sampling_check.exists() else None
    report = validate(a.grid_dir, a.golden, sampling, a.extract_dir)
    out = a.out or a.grid_dir / "validation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    for c in report["checks"]:
        line = f"{c['status'].upper():8} {c['name']}"
        if not a.summary_only and c["detail"]:
            line += f"  {c['detail']}"
        print(line)
    print(f"\n{'PASS' if report['passed'] else 'FAIL'}: {len(report['failures'])} failing, {len(report['skipped'])} skipped -> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
