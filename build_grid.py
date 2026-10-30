"""Stage B: build the exposure grid from extracted day files.

    uv run python build_grid.py --extract-dir out/extract --out out/grid

For every input day: segment legs, attribute them to airports and runways, rasterise onto H3
res 9, aggregate per cell, and compute receiver coverage and airport movement counts. Each
day's result is cached under ``<cache>/v<BUILDER_VERSION>/<date>/`` and reused while the
extract file, the reference data and BUILDER_VERSION are unchanged. The days are then merged
into ``<out>/<window_end>/{cells,coverage,airports}.parquet`` plus ``meta.json``.

Bump BUILDER_VERSION whenever a change here, or in segment/attribute/rasterise/aggregate/
coverage, can alter any output number: it invalidates every cached day.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date as Date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from h3.api import basic_int as h3i

import coverage as cov
from aggregate import DayAgg, aggregate_day, merge_frames
from attribute import attribute_aircraft
from grid_schema import (
    AIRPORTS_SCHEMA,
    COVERAGE_RES,
    GRID_SCHEMA_VERSION,
    MAX_COUNTED_ALT_FT,
    META_REQUIRED_KEYS,
    STATUS_GOOD,
    STATUS_INSUFFICIENT,
    STATUS_PARTIAL,
    conform,
)
from rasterise import rasterise_samples
from reference import DATA_DIR, Reference, build_reference, download_ourairports
from segment import Point, resample, split_legs

BUILDER_VERSION = 1
METHOD_URL = "https://github.com/eavesdrop-admin/uk-flight-grid"
CAPTURE_MIN = 0.85  # airports below this are under-captured; cells they dominate are downgraded
# Emitter categories that stand in for DfT "air transport movements". The plan says A3 to A5, but A2
# (15,500 to 75,000 lb) holds the ATR/Dash 8 turboprops that regional airports (Southampton, Norwich,
# Exeter) depend on; A3-only gave them capture ratios of 0.2 to 0.4 and downgraded their cells for no reason.
CAPTURE_CATEGORIES = ("A2", "A3", "A4", "A5")
SOURCE_REPO = "adsblol/globe_history"

CACHE_FILES = ("cells", "alt", "attrib", "cov_stats", "cov_hist", "moves")


# ------------------------------------------------------------- reference


def load_reference(ref_dir: Path) -> tuple[Reference, dict]:
    """OurAirports (downloaded once into ref_dir) joined to the vendored DfT figures."""
    airports, runways = ref_dir / "airports.csv", ref_dir / "runways.csv"
    if not (airports.exists() and runways.exists()):
        download_ourairports(ref_dir)
    ref = build_reference(airports, runways, regions=None)
    info = {
        "airports_csv_sha256": hashlib.sha256(airports.read_bytes()).hexdigest(),
        "runways_csv_sha256": hashlib.sha256(runways.read_bytes()).hexdigest(),
        "airports": len(ref.airports),
        "runway_ends": len(ref.runway_ends),
        "dft": ref.dft_source,
    }
    return ref, info


def reference_hash(ref: Reference) -> str:
    return ref.content_hash()


# ------------------------------------------------------------- one day


def _points(part: pl.DataFrame) -> list[Point]:
    t = (part["t_utc"].dt.epoch("ms").to_numpy() / 1000.0).tolist()
    cols = [part[c].to_list() for c in ("lat", "lon", "alt_ft", "gs", "track", "baro_rate", "leg_flag", "category", "mil", "heli", "alt_src")]
    return [
        Point(t=ti, lat=la, lon=lo, alt_ft=al, gs=gs, track=tr, baro_rate=br, leg_flag=lf, category=ca, mil=mi, heli=he, alt_src=src)
        for ti, la, lo, al, gs, tr, br, lf, ca, mi, he, src in zip(t, *cols)
    ]


def _concat(parts: list[np.ndarray], dtype) -> np.ndarray:
    return np.concatenate(parts) if parts else np.empty(0, dtype)


def process_day(df: pl.DataFrame, ref: Reference) -> tuple[DayAgg, cov.CoverageDay, pl.DataFrame]:
    """All per-day work: coverage statistics, leg attribution, rasterisation, aggregation."""
    coverage_day = cov.coverage_day(df)

    oh_parts: dict[str, list] = {k: [] for k in ("leg", "cell", "alt", "t", "inf")}
    nb_parts: dict[str, list] = {k: [] for k in ("leg", "cell", "alt", "t")}
    legs_rows: list[tuple[int, str, str, str]] = []
    moves: dict[tuple[str, str, str, bool], int] = {}
    leg_id = 0

    for part in df.sort("uid", "t_utc").partition_by("uid", maintain_order=True):
        legs = split_legs(_points(part))
        if not legs:
            continue
        for leg, att in zip(legs, attribute_aircraft(legs, ref)):
            if att.airport and att.label in ("A", "D", "C"):
                cat = next((p.category for p in leg.points if p.category), None)
                key = (att.airport, att.runway or "", att.label, cat in CAPTURE_CATEGORIES)
                moves[key] = moves.get(key, 0) + 1
            if min(p.alt_ft for p in leg.points) >= MAX_COUNTED_ALT_FT:  # type: ignore[type-var]
                continue
            res = resample(leg)
            r = rasterise_samples(res.samples, att.inferred_tail, fallback_t=leg.end_t)
            if len(r.overhead) == 0:
                continue
            for name, cs, parts in (("oh", r.overhead, oh_parts), ("nb", r.nearby, nb_parts)):
                parts["leg"].append(np.full(len(cs), leg_id, np.uint32))
                parts["cell"].append(cs.cells)
                parts["alt"].append(cs.alt_ft)
                parts["t"].append(cs.t)
                if name == "oh":
                    parts["inf"].append(cs.inferred)
            legs_rows.append((leg_id, att.airport or "", att.runway or "", att.label))
            leg_id += 1

    overhead = pl.DataFrame(
        {
            "leg_id": _concat(oh_parts["leg"], np.uint32),
            "cell": _concat(oh_parts["cell"], np.uint64),
            "alt_ft": _concat(oh_parts["alt"], np.float32),
            "t": _concat(oh_parts["t"], np.float64),
            "inferred": _concat(oh_parts["inf"], bool),
        }
    )
    nearby = pl.DataFrame(
        {
            "leg_id": _concat(nb_parts["leg"], np.uint32),
            "cell": _concat(nb_parts["cell"], np.uint64),
            "alt_ft": _concat(nb_parts["alt"], np.float32),
            "t": _concat(nb_parts["t"], np.float64),
        }
    )
    legs_df = pl.DataFrame(
        legs_rows, schema={"leg_id": pl.UInt32, "airport": pl.String, "runway": pl.String, "op": pl.String}, orient="row"
    )
    agg = aggregate_day(overhead, nearby, legs_df)
    moves_df = pl.DataFrame(
        [(*k, n) for k, n in moves.items()],
        schema={"airport": pl.String, "runway": pl.String, "op": pl.String, "large": pl.Boolean, "n": pl.UInt32},
        orient="row",
    )
    return agg, coverage_day, moves_df


# ----------------------------------------------------------------- cache


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(extract_sha: str, ref_hash: str) -> dict:
    return {"builder_version": BUILDER_VERSION, "extract_sha256": extract_sha, "reference_hash": ref_hash}


def day_cache_dir(cache_root: Path, day: str) -> Path:
    return cache_root / f"v{BUILDER_VERSION}" / day


def cache_valid(d: Path, key: dict) -> bool:
    try:
        return json.loads((d / "key.json").read_text()) == key and all((d / f"{n}.parquet").exists() for n in CACHE_FILES)
    except (OSError, ValueError):
        return False


def write_day_cache(d: Path, key: dict, agg: DayAgg, cday: cov.CoverageDay, moves: pl.DataFrame) -> None:
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name, frame in (
        ("cells", agg.cells), ("alt", agg.alt_hist), ("attrib", agg.attrib),
        ("cov_stats", cday.stats), ("cov_hist", cday.hist), ("moves", moves),
    ):
        frame.write_parquet(tmp / f"{name}.parquet", compression="zstd")
    (tmp / "key.json").write_text(json.dumps(key, sort_keys=True))
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)


def scan_day_caches(dirs: list[Path], name: str) -> list[pl.LazyFrame]:
    """Lazy scans of one cached table across days; the cell tables are too big to hold all at once."""
    return [pl.scan_parquet(d / f"{name}.parquet") for d in dirs]


def _worker(extract_path: str, cache_dir: str, ref_dir: str, key: dict) -> tuple[str, float]:
    t0 = time.time()
    ref, _ = load_reference(Path(ref_dir))
    agg, cday, moves = process_day(pl.read_parquet(extract_path), ref)
    write_day_cache(Path(cache_dir), key, agg, cday, moves)
    return Path(extract_path).stem, time.time() - t0


# ------------------------------------------------------------- airports


def build_airports(moves: pl.DataFrame, ref: Reference, n_days: int) -> pl.DataFrame:
    """One row per airport that has an observed movement or a DfT figure."""
    annual = 365.0 / n_days
    ad = moves.filter(pl.col("op").is_in(["A", "D"]))
    rows = []
    for icao, a in ref.airports.items():
        mine = ad.filter(pl.col("airport") == icao)
        observed = int(round(mine.filter(pl.col("large"))["n"].sum() * annual)) if mine.height else 0
        if not mine.height and a.dft_atm is None:
            continue
        resolved = mine.filter(pl.col("runway") != "")
        total = resolved["n"].sum() if resolved.height else 0
        top = []
        if total:
            g = resolved.group_by("runway", "op").agg(pl.col("n").sum()).sort(["n", "runway", "op"], descending=[True, False, False]).head(6)
            top = [{"runway": r, "op": o, "share": round(n / total, 3)} for r, o, n in g.iter_rows()]
        rows.append(
            {
                "icao": icao,
                "name": a.name,
                "observed_movements_annualised": observed,
                "dft_atm": a.dft_atm,
                "dft_year": a.dft_year,
                "capture_ratio": (observed / a.dft_atm) if a.dft_atm else None,
                "top_runways": json.dumps(top, separators=(",", ":")),
            }
        )
    return conform(pl.DataFrame(rows, schema_overrides={"dft_atm": pl.Int64, "dft_year": pl.Int64, "capture_ratio": pl.Float64}), AIRPORTS_SCHEMA).sort("icao")


def dependent_r7(cells: pl.DataFrame, airports: list[str]) -> list[int]:
    """Res-7 cells that contain a res-9 cell whose dominant overhead source is one of `airports`."""
    sel = cells.filter(pl.col("top1_airport").is_in(airports))["h3_r9"].to_list()
    return sorted({h3i.cell_to_parent(c, COVERAGE_RES) for c in sel})


# ----------------------------------------------------------------- main


def _gcd_stride(days: list[str]) -> int:
    ds = sorted(Date.fromisoformat(d) for d in days)
    g = 0
    for a, b in zip(ds, ds[1:]):
        g = math.gcd(g, (b - a).days)
    return g or 1


def discover_days(extract_dir: Path, only: list[str] | None) -> dict[str, Path]:
    found = {p.stem: p for p in sorted(extract_dir.rglob("*.parquet"))}
    if only:
        missing = [d for d in only if d not in found]
        if missing:
            raise SystemExit(f"no extract file for: {', '.join(missing)}")
        found = {d: found[d] for d in only}
    return found


def build(
    extract_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    ref_dir: Path,
    days: list[str] | None = None,
    day_stride: int | None = None,
    workers: int = 4,
    manifest_path: Path | None = None,
    method_url: str = METHOD_URL,
    min_sample_points: int = cov.MIN_SAMPLE_POINTS,
    log=print,
) -> Path:
    t_start = time.time()
    ref, ref_info = load_reference(ref_dir)
    ref_hash = reference_hash(ref)

    available = discover_days(extract_dir, days)
    manifest = {}
    mp = manifest_path or extract_dir.parent / "days_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text()).get("days", {})
    degraded = sorted(d for d in available if manifest.get(d, {}).get("status") == "degraded")
    used = {d: p for d, p in available.items() if d not in degraded}
    if not used:
        raise SystemExit("no usable days")
    names = sorted(used)
    stride = day_stride or _gcd_stride(names)

    keys = {d: cache_key(file_sha256(p), ref_hash) for d, p in used.items()}
    todo = [d for d in names if not cache_valid(day_cache_dir(cache_dir, d), keys[d])]
    log(f"{len(names)} days, {len(names) - len(todo)} cached, {len(todo)} to process (builder v{BUILDER_VERSION})")
    if todo:
        for d in todo:
            day_cache_dir(cache_dir, d).parent.mkdir(parents=True, exist_ok=True)
        if workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
                futs = [ex.submit(_worker, str(used[d]), str(day_cache_dir(cache_dir, d)), str(ref_dir), keys[d]) for d in todo]
                for f in futs:
                    name, secs = f.result()
                    log(f"  {name}: {secs:.0f}s")
        else:
            for d in todo:
                name, secs = _worker(str(used[d]), str(day_cache_dir(cache_dir, d)), str(ref_dir), keys[d])
                log(f"  {name}: {secs:.0f}s")

    dirs = [day_cache_dir(cache_dir, d) for d in names]
    n_days = len(names)

    t0 = time.time()
    cells = merge_frames(scan_day_caches(dirs, "cells"), scan_day_caches(dirs, "alt"), scan_day_caches(dirs, "attrib"))
    coverage = cov.finalise(
        [cov.CoverageDay(pl.read_parquet(d / "cov_stats.parquet"), pl.read_parquet(d / "cov_hist.parquet")) for d in dirs],
        min_sample_points=min_sample_points,
    )
    moves = pl.concat(scan_day_caches(dirs, "moves")).group_by("airport", "runway", "op", "large").agg(pl.col("n").sum()).collect()
    airports = build_airports(moves, ref, n_days)

    under = airports.filter(pl.col("capture_ratio") < CAPTURE_MIN)["icao"].to_list()
    if under:
        coverage = cov.downgrade(coverage, dependent_r7(cells, under))
    log(f"merge: {time.time() - t0:.0f}s")

    window_start, window_end = names[0], names[-1]
    out = out_dir / window_end
    out.mkdir(parents=True, exist_ok=True)
    cells.write_parquet(out / "cells.parquet", compression="zstd")
    coverage.write_parquet(out / "coverage.parquet", compression="zstd")
    airports.write_parquet(out / "airports.parquet", compression="zstd")

    status_counts = {s: int((coverage["status"] == s).sum()) for s in (STATUS_GOOD, STATUS_PARTIAL, STATUS_INSUFFICIENT)}
    meta = {
        "build_id": f"{window_end}-v{BUILDER_VERSION}-{hashlib.sha256(json.dumps([names, ref_hash, BUILDER_VERSION]).encode()).hexdigest()[:8]}",
        "window_start": window_start,
        "window_end": window_end,
        "days_observed": n_days,
        "days_degraded": len(degraded),
        "day_stride": stride,
        "builder_version": BUILDER_VERSION,
        "reference_hash": ref_hash,
        "method_url": method_url,
        # beyond the plan's flight_grid_meta columns
        "grid_schema_version": GRID_SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_days": names,
        "degraded_days": degraded,
        "data_vintage": {
            d: {k: manifest.get(d, {}).get(k) for k in ("release", "release_kind", "repo", "extracted_at", "points_kept")}
            for d in names
        },
        "source": {"name": "adsb.lol", "repo": SOURCE_REPO + "_<year>", "licence": "ODbL-1.0"},
        "reference": ref_info,
        "coverage": {
            "resolution": COVERAGE_RES,
            "min_sample_points": min_sample_points,
            "status_counts": status_counts,
            "capture_min": CAPTURE_MIN,
            "airports_below_capture_min": sorted(under),
        },
        "rows": {"cells": cells.height, "coverage": coverage.height, "airports": airports.height},
        "note": "Missing coverage rows (no observed point in the res-7 cell) mean insufficient, never zero flights.",
    }
    missing = [k for k in META_REQUIRED_KEYS if k not in meta]
    assert not missing, missing
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    log(f"wrote {out} in {time.time() - t_start:.0f}s: {cells.height} cells, {coverage.height} coverage rows {status_counts}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract-dir", type=Path, default=Path("out/extract"))
    ap.add_argument("--out", type=Path, default=Path("out/grid"))
    ap.add_argument("--cache", type=Path, default=Path("out/agg"))
    ap.add_argument("--reference-dir", type=Path, default=DATA_DIR / "raw" / "ourairports")
    ap.add_argument("--days", help="comma-separated YYYY-MM-DD subset of the extract directory")
    ap.add_argument("--day-stride", type=int, help="spacing of the sampled days (default: inferred from the dates)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--method-url", default=METHOD_URL)
    ap.add_argument("--min-sample-points", type=int, default=cov.MIN_SAMPLE_POINTS,
                    help="observed points a res-7 ring needs to be good/partial; lower it only for small test builds")
    a = ap.parse_args(argv)
    build(
        a.extract_dir, a.out, a.cache, a.reference_dir,
        days=a.days.split(",") if a.days else None, day_stride=a.day_stride,
        workers=a.workers, manifest_path=a.manifest, method_url=a.method_url,
        min_sample_points=a.min_sample_points,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
