"""Validation harness logic on small synthetic grids (no network, no real archive)."""

import json
from pathlib import Path

import polars as pl
import pytest
from h3.api import basic_int as h3i

import validate_grid as vg
from geo import along_cross_m
from grid_schema import AIRPORTS_SCHEMA, CELLS_SCHEMA, COVERAGE_SCHEMA, conform
from reference import build_reference

FIX = Path(__file__).parent / "fixtures"
LAT, LON = 51.4652, -0.3487


def cell_row(lat=LAT, lon=LON, **kw):
    row = {c: None for c in CELLS_SCHEMA}
    row.update(
        h3_r9=h3i.latlng_to_cell(lat, lon, 9), oh_day_mean=40.0, oh_day_p90=60, oh_lt1000=10.0, oh_1000_3000=30.0,
        oh_3000_7000=0.0, oh_night_mean=2.0, oh_alt_p50_ft=1100, oh_days_active=60, oh_inferred_share=0.0,
        nb_day_mean=80.0, nb_day_p90=100, nb_lt1000=20.0, nb_1000_3000=60.0, nb_3000_7000=0.0, nb_night_mean=4.0,
        top1_airport="EGLL", top1_runway="27L", top1_op="A", top1_share=70,
    )
    row.update(kw)
    return row


def cells_df(*rows):
    if not rows:
        return pl.DataFrame(schema=CELLS_SCHEMA)
    return conform(pl.DataFrame(list(rows), schema_overrides={c: pl.Int64 if t in (pl.Int16,) else t for c, t in CELLS_SCHEMA.items()}), CELLS_SCHEMA)


def coverage_df(lat=LAT, lon=LON, status="good", floor=900, extra=()):
    rows = [{"h3_r7": h3i.latlng_to_cell(lat, lon, 7), "status": status, "floor_ft": floor, "continuity": 0.95, "sample_points": 900}, *extra]
    return conform(pl.DataFrame(rows), COVERAGE_SCHEMA)


def point(**expect):
    return {"name": "p", "lat": LAT, "lon": LON, "expect": expect}


def run_point(p, cells, coverage, days=100):
    return vg.check_golden_point(p, vg.lookup_point(cells, coverage, p["lat"], p["lon"]), days)


# -------------------------------------------------------------- golden points


def test_golden_point_passes_when_everything_matches():
    p = point(top1_airport="EGLL", top1_op="A", top1_runway="27L", oh_day_mean_min=20, oh_alt_p50_ft=[800, 1500],
              days_active_frac=[0.5, 0.7], coverage_in=["good"])
    assert run_point(p, cells_df(cell_row()), coverage_df()).status == "pass"


@pytest.mark.parametrize(
    "overrides,expect_key,fragment",
    [
        ({"top1_runway": "27R"}, "top1_runway", "top runway 27R"),
        ({"top1_airport": "EGKK"}, "top1_airport", "top airport EGKK"),
        ({"top1_op": "D"}, "top1_op", "top op D"),
        ({"oh_alt_p50_ft": 2500}, "oh_alt_p50_ft", "median altitude 2500"),
        ({"oh_days_active": 100}, "days_active_frac", "active 1.00"),
        ({"oh_day_mean": 3.0}, "oh_day_mean_min", "3.0 legs a day, expected at least"),
    ],
)
def test_golden_point_fails_on_each_mismatch(overrides, expect_key, fragment):
    exp = {"top1_airport": "EGLL", "top1_op": "A", "top1_runway": "27L", "oh_day_mean_min": 20,
           "oh_alt_p50_ft": [800, 1500], "days_active_frac": [0.5, 0.7]}
    res = run_point(point(**exp), cells_df(cell_row(**overrides)), coverage_df())
    assert res.status == "fail" and fragment in res.detail


def test_runway_and_op_accept_lists():
    p = point(top1_op=["A", "D"], top1_runway=["27L", "27R"])
    assert run_point(p, cells_df(cell_row(top1_op="D", top1_runway="27R")), coverage_df()).status == "pass"


def test_absent_cell_fails_a_traffic_minimum_and_a_missing_row_is_not_zero_flights():
    res = run_point(point(oh_day_mean_min=1), cells_df(cell_row(lat=53.0, lon=-2.0)), coverage_df())
    assert res.status == "fail" and "no cell row" in res.detail


def test_control_point_may_have_no_cell_but_not_an_airport():
    ctrl = point(no_airport_attribution=True, cell_may_be_absent=True, oh_day_mean_max=10)
    assert run_point(ctrl, cells_df(cell_row(lat=53.0, lon=-2.0)), coverage_df()).status == "pass"
    assert "expected no airport" in run_point(ctrl, cells_df(cell_row(oh_day_mean=5.0)), coverage_df()).detail
    assert "at most 10" in run_point(ctrl, cells_df(cell_row(oh_day_mean=50.0, top1_airport=None)), coverage_df()).detail


def test_missing_coverage_row_counts_as_insufficient():
    p = point(cell_may_be_absent=True, coverage_in=["good"])
    res = run_point(p, cells_df(cell_row(lat=53.0, lon=-2.0)), coverage_df(lat=53.0, lon=-2.0, status="good", floor=900).filter(pl.lit(False)))
    assert res.status == "fail" and "coverage insufficient" in res.detail


def test_good_coverage_with_high_floor_fails_the_poor_coverage_control():
    p = point(cell_may_be_absent=True, not_good_unless_floor_le=1500)
    assert run_point(p, cells_df(), coverage_df(status="good", floor=2500)).status == "fail"
    assert run_point(p, cells_df(), coverage_df(status="partial_low_altitude", floor=2500)).status == "pass"
    assert run_point(p, cells_df(), coverage_df(status="good", floor=1200)).status == "pass"


# ------------------------------------------------------------- systematic checks


def airports_df(ratios: dict[str, float | None], dft=100_000):
    rows = [{"icao": k, "name": k, "observed_movements_annualised": 1, "dft_atm": dft - i, "dft_year": 2024,
             "capture_ratio": v, "top_runways": "[]"} for i, (k, v) in enumerate(ratios.items())]
    return conform(pl.DataFrame(rows, schema_overrides={"capture_ratio": pl.Float64}), AIRPORTS_SCHEMA)


EW = ["EGLL", "EGKK", "EGCC", "EGSS", "EGGW", "EGBB", "EGGD", "EGNX", "EGNT", "EGGP", "EGNM", "EGFF", "EGHI", "EGHH", "EGNV", "EGSH"]


def test_capture_ratio_passes_when_top_fifteen_are_in_range():
    res = vg.check_capture_ratio(airports_df({k: 1.0 for k in EW}))
    assert res.status == "pass"


def test_capture_ratio_fails_below_and_above_the_band_and_when_null():
    base = {k: 1.0 for k in EW}
    for bad in (0.89, 1.36, None):
        res = vg.check_capture_ratio(airports_df({**base, "EGKK": bad}))
        assert res.status == "fail" and "EGKK" in res.detail


def test_bournemouth_and_norwich_are_exempt_from_the_upper_bound_only():
    base = {k: 1.0 for k in EW}
    res = vg.check_capture_ratio(airports_df({**base, "EGHH": 1.8}))
    assert res.status == "pass"
    assert next(a for a in res.data["airports"] if a["icao"] == "EGHH")["upper_bound_exempt"]
    assert vg.check_capture_ratio(airports_df({**base, "EGHH": 0.5})).status == "fail"
    assert vg.check_capture_ratio(airports_df({**base, "EGKK": 1.8})).status == "fail"
    top = {k: 1.0 for k in EW if k != "EGNV"}  # puts EGSH in the top fifteen
    assert vg.check_capture_ratio(airports_df({**top, "EGSH": 1.6})).status == "pass"
    assert vg.check_capture_ratio(airports_df({**top, "EGSH": 0.5})).status == "fail"


def test_capture_ratio_only_counts_england_and_wales():
    # A Scottish airport with a huge DfT figure must not take a top-15 slot, and a bad ratio there is ignored.
    ratios = {"EGPH": 0.1, **{k: 1.0 for k in EW[:15]}}
    res = vg.check_capture_ratio(airports_df(ratios))
    assert res.status == "pass"
    assert not any(a["icao"] == "EGPH" for a in res.data["airports"])


def test_capture_ratio_fails_if_fewer_than_fifteen_airports_have_a_dft_figure():
    assert vg.check_capture_ratio(airports_df({k: 1.0 for k in EW[:10]})).status == "fail"


def test_is_england_wales():
    assert vg.is_england_wales("EGLL") and vg.is_england_wales("EGFF")
    assert not any(vg.is_england_wales(x) for x in ("EGPH", "EGPF", "EGAA", "EGQL", "LFPG"))


def test_coverage_invariant_and_empty_table():
    ok = coverage_df()
    assert vg.check_coverage(ok).status == "pass"
    assert vg.check_coverage(coverage_df(status="good", floor=1600)).status == "fail"
    assert vg.check_coverage(ok.filter(pl.lit(False))).status == "fail"
    only_insufficient = coverage_df(status="insufficient", floor=3000)
    assert "no 'good' cells" in vg.check_coverage(only_insufficient).detail


def test_coverage_reports_insufficient_share_without_gating_it():
    extra = [{"h3_r7": h3i.latlng_to_cell(53, -2, 7), "status": "insufficient", "floor_ft": 5000, "continuity": 0.1, "sample_points": 3}]
    res = vg.check_coverage(coverage_df(extra=extra))
    assert res.status == "pass" and res.data["insufficient_share"] == 0.5


def test_days_gate_matches_the_loader_numbers():
    assert vg.required_days(1) == 300 and vg.required_days(4) == 80 and vg.required_days(2) == 150
    assert vg.check_days_gate({"day_stride": 4, "days_observed": 80}).status == "pass"
    assert vg.check_days_gate({"day_stride": 4, "days_observed": 79}).status == "fail"
    assert vg.check_days_gate({"day_stride": 1, "days_observed": 299}).status == "fail"


def test_sampling_check_rules():
    meta = {"day_stride": 4, "builder_version": 1}
    good = {"passed": True, "builder_version": 1, "stride": 4, "month": "2026-09", "tolerance": 0.1}
    assert vg.check_sampling(meta, good).status == "pass"
    assert vg.check_sampling(meta, None).status == "fail"
    assert vg.check_sampling(meta, {**good, "passed": False}).status == "fail"
    assert "builder v2" in vg.check_sampling(meta, {**good, "builder_version": 2}).detail
    assert "stride 2" in vg.check_sampling(meta, {**good, "stride": 2}).detail
    assert vg.check_sampling({"day_stride": 1, "builder_version": 1}, None).status == "skipped"


def test_baro_share(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def day(name, srcs, alts):
        (tmp_path / "2026").mkdir(exist_ok=True)
        pq.write_table(pa.table({"alt_ft": alts, "alt_src": srcs}), tmp_path / "2026" / f"{name}.parquet")

    day("2026-01-01", ["geom"] * 9 + ["baro"], [500] * 10)
    assert vg.check_baro_share(tmp_path, ["2026-01-01"]).status == "pass"  # 10% < 20%
    day("2026-01-02", ["baro"] * 5 + ["geom"] * 5, [500] * 10)
    over = vg.check_baro_share(tmp_path, ["2026-01-01", "2026-01-02"])
    assert over.status == "fail" and over.data["baro_share"] == 0.3  # 6 of 20
    # Points at or above 7,000 ft are not counted, so they cannot dilute or inflate the share.
    day("2026-01-03", ["baro"] * 10, [9000] * 10)
    assert vg.check_baro_share(tmp_path, ["2026-01-01", "2026-01-03"]).data["baro_share"] == 0.1
    assert vg.check_baro_share(tmp_path, ["2026-01-09"]).status == "fail"  # missing extract


# ------------------------------------------------------------------ end to end


def write_grid(d: Path, *, days=100, airports=None, cells=None, coverage=None, stride=4):
    d.mkdir(parents=True, exist_ok=True)
    (cells if cells is not None else cells_df(cell_row())).write_parquet(d / "cells.parquet")
    (coverage if coverage is not None else coverage_df()).write_parquet(d / "coverage.parquet")
    (airports if airports is not None else airports_df({k: 1.0 for k in EW})).write_parquet(d / "airports.parquet")
    (d / "meta.json").write_text(json.dumps({
        "build_id": "b1", "window_start": "2025-10-01", "window_end": "2026-09-30", "days_observed": days,
        "day_stride": stride, "builder_version": 1, "input_days": [],
    }))


def golden_file(tmp_path):
    p = tmp_path / "golden.json"
    p.write_text(json.dumps({"points": [{"name": "t", "lat": LAT, "lon": LON, "expect": {
        "top1_airport": "EGLL", "top1_op": "A", "oh_day_mean_min": 20, "coverage_in": ["good"]}}]}))
    return p


SAMPLING = {"passed": True, "builder_version": 1, "stride": 4, "month": "2026-09", "tolerance": 0.1}


def test_validate_passes_and_writes_the_keys_the_loader_reads(tmp_path):
    write_grid(tmp_path / "g")
    code = vg.main([str(tmp_path / "g"), "--golden", str(golden_file(tmp_path)), "--sampling-check", str(_dump(tmp_path, SAMPLING))])
    v = json.loads((tmp_path / "g" / "validation.json").read_text())
    assert code == 0
    assert v["passed"] is True and v["status"] == "pass" and v["sampling_check_passed"] is True
    assert v["failures"] == [] and v["skipped"] == ["baro_share"]
    assert v["deferred"] and v["build_id"] == "b1"
    assert vg.DEFERRED[0].startswith("spearman")


def _dump(tmp_path, obj):
    p = tmp_path / "sampling_check.json"
    p.write_text(json.dumps(obj))
    return p


def test_validate_refuses_a_failing_build_with_nonzero_exit_but_still_writes_the_report(tmp_path):
    write_grid(tmp_path / "g", days=3)
    code = vg.main([str(tmp_path / "g"), "--golden", str(golden_file(tmp_path)), "--sampling-check", str(_dump(tmp_path, SAMPLING))])
    v = json.loads((tmp_path / "g" / "validation.json").read_text())
    assert code == 1
    assert v["passed"] is False and v["status"] == "fail" and v["failures"] == ["days_gate"]


def test_validate_without_a_pilot_check_fails_a_sampled_build_and_flags_it(tmp_path):
    write_grid(tmp_path / "g")
    code = vg.main([str(tmp_path / "g"), "--golden", str(golden_file(tmp_path))])
    v = json.loads((tmp_path / "g" / "validation.json").read_text())
    assert code == 1 and v["sampling_check_passed"] is False and v["failures"] == ["sampling_check"]


def test_full_build_needs_no_pilot_check(tmp_path):
    write_grid(tmp_path / "g", days=300, stride=1)
    assert vg.main([str(tmp_path / "g"), "--golden", str(golden_file(tmp_path))]) == 0


def test_summary_only_prints_no_values(tmp_path, capsys):
    write_grid(tmp_path / "g", days=3)
    vg.main([str(tmp_path / "g"), "--golden", str(golden_file(tmp_path)), "--summary-only"])
    out = capsys.readouterr().out
    assert "FAIL     days_gate" in out and "days observed" not in out


# -------------------------------------------------------------- the golden file


GOLDEN = json.loads((Path(vg.__file__).parent / "golden_points.json").read_text())["points"]


def test_golden_file_is_well_formed():
    names = [p["name"] for p in GOLDEN]
    assert len(names) == len(set(names)) >= 12
    for p in GOLDEN:
        assert 49.5 < p["lat"] < 56.0 and -7.0 < p["lon"] < 2.2 and p["expect"]
    covered = {p["centreline"]["airport"] for p in GOLDEN if "centreline" in p}
    assert {"EGLL", "EGKK", "EGCC", "EGGD", "EGFF"} <= covered  # Heathrow, Gatwick, Manchester, Bristol, Wales
    controls = [p for p in GOLDEN if p["expect"].get("no_airport_attribution")]
    assert len(controls) >= 3


@pytest.mark.parametrize("p", [p for p in GOLDEN if "centreline" in p], ids=lambda p: p["name"])
def test_golden_points_sit_on_the_extended_centreline(p):
    ref = build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", regions=None)
    end = ref.runway_end(p["centreline"]["airport"], p["centreline"]["runway"])
    if end is None:
        pytest.skip("runway not in the trimmed fixtures")
    along, cross = along_cross_m(end.thr_lat, end.thr_lon, end.heading, p["lat"], p["lon"])
    assert 2_500 < -along < 7_000, "point should be on the approach side, a few km out"
    assert abs(cross) < 150, "point should be on the centreline, not between glidepaths"
