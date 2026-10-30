"""End to end on tiny synthetic days: extract Parquet in, grid Parquet and meta.json out."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from h3.api import basic_int as h3i

import build_grid
from grid_schema import AIRPORTS_SCHEMA, CELLS_SCHEMA, COVERAGE_SCHEMA, META_REQUIRED_KEYS
from reference import build_reference
from schema import COLUMNS, PARQUET_SCHEMA
from synth import final_approach

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ref():
    return build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", regions=None)


def write_day(path: Path, date: str, hours: list[int], ref, uid0: int):
    """One straight-in 27L approach per hour (UTC) on `date`, one aircraft each."""
    end = {r.ident: r for r in ref.airports["EGLL"].runways}["27L"]
    rows = []
    for i, h in enumerate(hours):
        t0 = datetime.fromisoformat(f"{date}T{h:02d}:00:00+00:00").timestamp()
        for p in final_approach(end, 14, 100, t0=t0):
            rows.append(
                {
                    "uid": uid0 + i, "leg_flag": p.leg_flag, "t_utc": datetime.fromtimestamp(p.t, timezone.utc),
                    "lat": p.lat, "lon": p.lon, "alt_ft": round(p.alt_ft), "alt_src": "geom", "gs": float(p.gs),
                    "track": float(p.track), "baro_rate": round(p.baro_rate), "category": "A3", "mil": False, "heli": False,
                }
            )
    df = pl.DataFrame(rows).cast({"uid": pl.UInt64, "t_utc": pl.Datetime("ms", "UTC")})
    df = df.select(COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(df.to_arrow().cast(PARQUET_SCHEMA), path)
    return end


@pytest.fixture()
def world(tmp_path, ref):
    refdir = tmp_path / "ref"
    refdir.mkdir()
    shutil.copy(FIX / "airports_trim.csv", refdir / "airports.csv")
    shutil.copy(FIX / "runways_trim.csv", refdir / "runways.csv")
    ex = tmp_path / "extract"
    # 2026-11-10 (GMT): three approaches, one at 02:00 UTC (night). 2026-11-14: one approach at noon.
    end = write_day(ex / "2026" / "2026-11-10.parquet", "2026-11-10", [2, 10, 14], ref, 100)
    write_day(ex / "2026" / "2026-11-14.parquet", "2026-11-14", [12], ref, 200)
    return tmp_path, refdir, ex, end


def run(world, **kw):
    tmp, refdir, ex, _ = world
    logs: list[str] = []
    out = build_grid.build(ex, tmp / "grid", tmp / "agg", refdir, workers=1, log=logs.append, **kw)
    return out, logs


def test_two_days_build_matches_schema_and_arithmetic(world):
    out, _ = run(world)
    end = world[3]
    cells = pl.read_parquet(out / "cells.parquet")
    cov = pl.read_parquet(out / "coverage.parquet")
    airports = pl.read_parquet(out / "airports.parquet")
    assert dict(cells.schema) == CELLS_SCHEMA
    assert dict(cov.schema) == COVERAGE_SCHEMA
    assert dict(airports.schema) == AIRPORTS_SCHEMA

    # a cell on the extended centreline about 4 km out (3 degree path: about 700 ft above the runway)
    from geo import destination

    lat, lon = destination(end.thr_lat, end.thr_lon, (end.heading + 180) % 360, 4000)
    c = cells.filter(pl.col("h3_r9") == h3i.latlng_to_cell(lat, lon, 9)).row(0, named=True)
    # day 1 sees 3 legs, day 2 sees 1: mean 2.0 over the two observed days
    assert c["oh_day_mean"] == pytest.approx(2.0)
    assert c["oh_days_active"] == 2
    assert c["oh_day_p90"] == 3  # percentile([3, 1], 90) = 2.8
    assert c["oh_night_mean"] == pytest.approx(0.5)  # one 02:00 GMT pass in two days
    assert c["oh_lt1000"] + c["oh_1000_3000"] + c["oh_3000_7000"] == pytest.approx(2.0)
    assert 500 < c["oh_alt_p50_ft"] < 2000
    assert (c["top1_airport"], c["top1_runway"], c["top1_op"], c["top1_share"]) == ("EGLL", "27L", "A", 100)
    assert c["top2_op"] is None
    assert c["nb_day_mean"] >= c["oh_day_mean"]


def test_meta_json_has_the_contract_and_the_provenance(world):
    out, _ = run(world)
    meta = json.loads((out / "meta.json").read_text())
    for k in META_REQUIRED_KEYS:
        assert k in meta
    assert (meta["window_start"], meta["window_end"]) == ("2026-11-10", "2026-11-14")
    assert meta["days_observed"] == 2 and meta["day_stride"] == 4 and meta["days_degraded"] == 0
    assert meta["builder_version"] == build_grid.BUILDER_VERSION
    assert len(meta["reference_hash"]) == 64
    assert meta["input_days"] == ["2026-11-10", "2026-11-14"]
    assert meta["rows"]["cells"] == pl.read_parquet(out / "cells.parquet").height
    assert out.name == "2026-11-14"


def test_airports_table_counts_movements_and_capture(world):
    out, _ = run(world)
    a = pl.read_parquet(out / "airports.parquet").filter(pl.col("icao") == "EGLL").row(0, named=True)
    # 4 arrivals in 2 days, annualised: 4 / 2 * 365
    assert a["observed_movements_annualised"] == 730
    assert a["dft_atm"] == 473_799 and a["dft_year"] == 2024
    assert a["capture_ratio"] == pytest.approx(730 / 473_799, rel=1e-3)
    assert json.loads(a["top_runways"]) == [{"runway": "27L", "op": "A", "share": 1.0}]


def test_unobserved_places_have_no_rows_not_zero_rows(world):
    out, _ = run(world)
    cov = pl.read_parquet(out / "coverage.parquet")
    far = h3i.latlng_to_cell(52.3, -3.5, 7)  # mid-Wales, no points in the synthetic days
    assert far not in cov["h3_r7"].to_list()
    assert set(cov["status"]) <= {"good", "partial_low_altitude", "insufficient"}


def test_day_cache_is_reused_and_invalidated_by_builder_version(world, monkeypatch):
    _, logs = run(world)
    assert "2 cached" not in logs[0] and "0 cached" in logs[0]
    _, logs = run(world)
    assert "2 cached, 0 to process" in logs[0]
    monkeypatch.setattr(build_grid, "BUILDER_VERSION", build_grid.BUILDER_VERSION + 1)
    _, logs = run(world)
    assert "0 cached, 2 to process" in logs[0]


def test_changed_extract_file_invalidates_only_that_day(world):
    tmp, refdir, ex, _ = world
    run(world)
    p = ex / "2026" / "2026-11-14.parquet"
    ref = build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", regions=None)
    write_day(p, "2026-11-14", [12, 13], ref, 300)
    _, logs = run(world)
    assert "1 cached, 1 to process" in logs[0]


def test_degraded_days_are_excluded_and_counted(world):
    tmp, _, ex, _ = world
    (tmp / "days_manifest.json").write_text(
        json.dumps({"days": {"2026-11-10": {"status": "ok"}, "2026-11-14": {"status": "degraded"}}})
    )
    out, _ = run(world, manifest_path=tmp / "days_manifest.json")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["input_days"] == ["2026-11-10"] and meta["days_degraded"] == 1 and meta["degraded_days"] == ["2026-11-14"]


def test_stride_is_inferred_from_the_dates():
    assert build_grid._gcd_stride(["2026-10-19", "2026-10-23", "2026-10-27"]) == 4
    assert build_grid._gcd_stride(["2026-10-19", "2026-10-21", "2026-10-27"]) == 2
    assert build_grid._gcd_stride(["2026-10-19"]) == 1


def test_capture_counts_airliner_categories_not_light_aircraft(ref):
    moves = pl.DataFrame(
        [("EGLL", "27L", "A", True, 6), ("EGLL", "27L", "A", False, 100), ("EGLL", "09R", "D", True, 2), ("EGLL", "", "C", True, 50)],
        schema={"airport": pl.String, "runway": pl.String, "op": pl.String, "large": pl.Boolean, "n": pl.UInt32},
        orient="row",
    )
    a = build_grid.build_airports(moves, ref, n_days=365).filter(pl.col("icao") == "EGLL").row(0, named=True)
    assert a["observed_movements_annualised"] == 8  # circuits and light aircraft are not air transport movements
    assert {"A2", "A3", "A4", "A5"} == set(build_grid.CAPTURE_CATEGORIES)
    runways = json.loads(a["top_runways"])  # shares use every resolved A/D movement, light aircraft included
    assert runways[0] == {"runway": "27L", "op": "A", "share": round(106 / 108, 3)}


def test_airports_without_movements_or_dft_figure_are_omitted(ref):
    empty = pl.DataFrame(schema={"airport": pl.String, "runway": pl.String, "op": pl.String, "large": pl.Boolean, "n": pl.UInt32})
    out = build_grid.build_airports(empty, ref, n_days=1)
    assert set(out["icao"]) == {i for i, a in ref.airports.items() if a.dft_atm is not None}
    assert out["observed_movements_annualised"].sum() == 0


def test_helicopter_movements_count_towards_capture_but_not_runway_shares(ref):
    moves = pl.DataFrame(
        [("EGLL", "27L", "A", True, 4), ("EGLL", "", "HA", True, 3), ("EGLL", "", "HD", True, 2)],
        schema={"airport": pl.String, "runway": pl.String, "op": pl.String, "large": pl.Boolean, "n": pl.UInt32},
        orient="row",
    )
    a = build_grid.build_airports(moves, ref, n_days=365).filter(pl.col("icao") == "EGLL").row(0, named=True)
    assert a["observed_movements_annualised"] == 9
    assert json.loads(a["top_runways"]) == [{"runway": "27L", "op": "A", "share": 1.0}]
