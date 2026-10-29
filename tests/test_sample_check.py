"""Sampled-vs-full comparison logic and the month driver, with the grid builder stubbed out."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from h3.api import basic_int as h3i

import sample_check as sc

GOLD = [{"name": "g", "lat": 51.4652, "lon": -0.3487}]
GOLD_CELL = h3i.latlng_to_cell(51.4652, -0.3487, 9)


def cells(means: dict[int, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"h3_r9": list(means), "oh_day_mean": [float(v) for v in means.values()],
         "oh_day_p90": [int(v * 2) for v in means.values()], "oh_days_active": [20] * len(means)},
        schema_overrides={"h3_r9": pl.Int64},
    )


def busy(n=1000, base=10_000_000_000):
    return {base + i: 100.0 - i * 0.05 for i in range(n)}


def test_identical_grids_pass():
    full = {**busy(), GOLD_CELL: 40.0}
    r = sc.compare(cells(full), cells(full), GOLD)
    assert r["passed"] and r["golden_gated"] == 1 and r["top_within_fraction"] == 1.0


def test_golden_point_outside_ten_percent_fails():
    full = {**busy(), GOLD_CELL: 40.0}
    r = sc.compare(cells(full), cells({**full, GOLD_CELL: 35.0}), GOLD)  # 12.5% off
    assert not r["passed"] and any("golden points outside" in x for x in r["reasons"])
    ok = sc.compare(cells(full), cells({**full, GOLD_CELL: 37.0}), GOLD)  # 7.5% off
    assert ok["passed"]


def test_tolerance_boundary_is_inclusive():
    full = {**busy(), GOLD_CELL: 40.0}
    assert sc.compare(cells(full), cells({**full, GOLD_CELL: 36.0}), GOLD)["passed"]  # exactly 10%


def test_top_cells_share_gate():
    full = busy()
    sampled = dict(full)
    for c in list(full)[:89]:  # 8.9% of the top 1000 far off: still passes
        sampled[c] = full[c] * 2
    near = sc.compare(cells({**full, GOLD_CELL: 40.0}), cells({**sampled, GOLD_CELL: 40.0}), GOLD)
    assert near["passed"] or near["top_within_fraction"] >= 0.9
    for c in list(full)[:200]:  # 20% far off: fails
        sampled[c] = full[c] * 2
    bad = sc.compare(cells({**full, GOLD_CELL: 40.0}), cells({**sampled, GOLD_CELL: 40.0}), GOLD)
    assert not bad["passed"] and any("busiest cells" in x for x in bad["reasons"])


def test_cell_missing_from_the_sample_counts_as_zero():
    full = {**busy(10), GOLD_CELL: 40.0}
    sampled = {k: v for k, v in full.items() if k != GOLD_CELL}
    r = sc.compare(cells(full), cells(sampled), GOLD)
    assert not r["passed"] and r["golden"][0]["sampled_oh_day_mean"] == 0.0


def test_quiet_golden_points_are_reported_not_gated_and_at_least_one_must_qualify():
    full = {**busy(10), GOLD_CELL: 0.4}
    r = sc.compare(cells(full), cells({**full, GOLD_CELL: 0.9}), GOLD)
    assert r["golden"][0]["gated"] is False
    assert not r["passed"] and any("no golden point" in x for x in r["reasons"])


def test_golden_reports_p90_and_days_active_without_gating_them():
    full = {**busy(10), GOLD_CELL: 40.0}
    r = sc.compare(cells(full), cells(full), GOLD)
    g = r["golden"][0]
    assert {"full_oh_day_p90", "sampled_oh_day_p90", "full_days_active", "sampled_days_active"} <= set(g)


def test_empty_full_grid_fails():
    r = sc.compare(cells({}).clear(), cells({}).clear(), GOLD)
    assert not r["passed"]


# ------------------------------------------------------------------- the driver


def fake_build_factory(calls):
    def fake_build(extract_dir, out_dir, cache_dir, ref_dir, days=None, day_stride=None, workers=4, log=print, **_):
        calls.append((list(days), day_stride))
        out = Path(out_dir) / days[-1]
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps({"input_days": list(days)}))
        cells({**busy(50), GOLD_CELL: 40.0}).write_parquet(out / "cells.parquet")
        return out
    return fake_build


@pytest.fixture
def month_extracts(tmp_path):
    d = tmp_path / "extract" / "2026"
    d.mkdir(parents=True)
    for day in range(1, 31):
        (d / f"2026-09-{day:02d}.parquet").write_bytes(b"")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"points": GOLD}))
    return tmp_path, golden


def test_run_builds_full_month_then_the_ordinal_stride_subset(month_extracts, monkeypatch):
    tmp, golden = month_extracts
    calls = []
    monkeypatch.setattr(sc.build_grid, "build", fake_build_factory(calls))
    res = sc.run("2026-09", tmp / "extract", tmp / "work", tmp / "cache", tmp / "ref", golden)
    full_days, full_stride = calls[0]
    sub_days, sub_stride = calls[1]
    assert len(full_days) == 30 and full_stride == 1 and sub_stride == 4
    assert sub_days == [d for d in full_days if date.fromisoformat(d).toordinal() % 4 == 0]
    assert res["passed"] and res["days_full"] == 30 and res["days_sampled"] == len(sub_days)
    assert res["builder_version"] == sc.build_grid.BUILDER_VERSION and res["stride"] == 4


def test_run_refuses_a_month_with_missing_days(month_extracts, monkeypatch):
    tmp, golden = month_extracts
    for p in list((tmp / "extract" / "2026").glob("2026-09-1*.parquet")):
        p.unlink()  # 10 of 30 days gone
    calls = []
    monkeypatch.setattr(sc.build_grid, "build", fake_build_factory(calls))
    res = sc.run("2026-09", tmp / "extract", tmp / "work", tmp / "cache", tmp / "ref", golden)
    assert not res["passed"] and "not a full month" in res["reasons"][0]
    assert len(calls) == 1  # never built the sampled grid
