"""Run the extractor over ~20 traces cut from a real day (2026-10-27)."""

from __future__ import annotations

import gzip
import io
import json
import math
import tarfile
from datetime import date
from pathlib import Path

import orjson
import pyarrow.parquet as pq
import pytest

import extract_day as ex
import schema
from remote_tar import SplitTarFile

FIXTURE = Path(__file__).parent / "fixtures" / "traces_2026-10-27.json"
DAY = date(2026, 10, 27)


@pytest.fixture(scope="module")
def traces() -> list[dict]:
    return orjson.loads(FIXTURE.read_bytes())["traces"]


def by_scenario(traces, name):
    return [t for t in traces if t["_scenario"] == name]


def test_fixture_has_about_twenty_traces_covering_each_scenario(traces):
    assert 18 <= len(traces) <= 24
    assert {t["_scenario"] for t in traces} >= {
        "helicopter", "military", "heathrow_arrival", "heathrow_departure", "ground_ops",
        "stale_points", "baro_only", "mlat_points", "cruise_only", "outside_uk", "multi_leg",
    }


def test_fixture_carries_no_identifying_fields(traces):
    allowed_top = {"icao", "timestamp", "trace", "dbFlags", "_scenario"}
    for t in traces:
        assert set(t) <= allowed_top
        assert t["icao"].startswith("f0")  # synthetic address
        for p in t["trace"]:
            assert p[8] is None or set(p[8]) == {"category"}


def test_fixture_exercises_the_awkward_point_types(traces):
    pts = [p for t in traces for p in t["trace"]]
    assert any(p[3] == "ground" for p in pts)
    assert any(p[6] & 1 for p in pts)            # stale
    assert any(p[6] & 2 for p in pts)            # new leg
    assert any(p[10] is None and isinstance(p[3], int) for p in pts)  # no alt_geom
    assert any(p[9] == "mlat" for p in pts)
    assert any((t.get("dbFlags") or 0) & 1 for t in traces)
    assert any(p[8] and p[8]["category"] == "A7" for t in traces for p in t["trace"])


def test_scenarios_produce_the_expected_output(traces, salt):
    for t in by_scenario(traces, "outside_uk") + by_scenario(traces, "cruise_only"):
        cols, stats = ex.extract_trace(t, salt)
        assert cols is None, t["icao"]
    for t in by_scenario(traces, "outside_uk"):
        assert ex.extract_trace(t, salt)[1].aircraft_uk == 0
    for t in by_scenario(traces, "helicopter"):
        cols, _ = ex.extract_trace(t, salt)
        assert cols and all(cols["heli"]) and set(cols["category"]) == {"A7"}
    for t in by_scenario(traces, "military"):
        cols, _ = ex.extract_trace(t, salt)
        assert cols and all(cols["mil"])
    for t in by_scenario(traces, "baro_only"):
        cols, _ = ex.extract_trace(t, salt)
        assert cols and set(cols["alt_src"]) == {"baro"}
    for t in by_scenario(traces, "multi_leg") + by_scenario(traces, "ground_ops"):
        cols, _ = ex.extract_trace(t, salt)
        assert cols and sum(cols["leg_flag"]) >= 2


def test_invariants_hold_for_every_kept_point(traces, salt):
    for t in traces:
        cols, stats = ex.extract_trace(t, salt)
        assert stats.points_total == len(t["trace"])
        if cols is None:
            continue
        n = len(cols["t_utc"])
        assert stats.points_kept == n
        assert cols["leg_flag"][0] is True
        assert all(
            schema.LAT_MIN <= a <= schema.LAT_MAX and schema.LON_MIN <= b <= schema.LON_MAX
            for a, b in zip(cols["lat"], cols["lon"])
        )
        assert all(schema.ALT_MIN_FT <= a <= schema.ALT_MAX_FT for a in cols["alt_ft"])
        assert all(g >= schema.MIN_GS_KT for g in cols["gs"])
        assert cols["t_utc"] == sorted(cols["t_utc"])
        # thinning: inside a leg, consecutive points are 8 s or 400 m apart
        for i in range(1, n):
            if cols["leg_flag"][i]:
                continue
            dt = (cols["t_utc"][i] - cols["t_utc"][i - 1]) / 1000
            dy = (cols["lat"][i] - cols["lat"][i - 1]) * ex.M_PER_DEG
            dx = (cols["lon"][i] - cols["lon"][i - 1]) * ex.M_PER_DEG * math.cos(math.radians(cols["lat"][i]))
            assert dt >= schema.THIN_MIN_DT_S or math.hypot(dx, dy) >= schema.THIN_MIN_DIST_M - 1e-6


def write_split_tar(tmp_path: Path, traces: list[dict], parts: int = 2) -> list[tuple[str, int]]:
    """A tar shaped like adsb.lol's: acas/, a large heatmap/ entry, then gzip traces/."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        add("./acas/acas.json.gz", b"acas")
        add("./heatmap/00.bin.ttf", b"\0" * (10 * 1024 * 1024))
        for t in traces:
            body = {k: v for k, v in t.items() if k != "_scenario"}
            add(f"./traces/{t['icao'][-2:]}/trace_full_{t['icao']}.json", gzip.compress(orjson.dumps(body)))
    blob = buf.getvalue()
    cut = -(-len(blob) // parts)
    out = []
    for i in range(parts):
        p = tmp_path / f"day.tar.a{chr(97 + i)}"
        p.write_bytes(blob[i * cut : (i + 1) * cut])
        out.append((str(p), p.stat().st_size))
    return out


@pytest.mark.parametrize("workers", [1, 2])
def test_end_to_end_parquet(tmp_path, traces, workers):
    parts = write_split_tar(tmp_path, traces)
    out = tmp_path / "out"
    entry = ex.extract_day(DAY, out, parts=parts, workers=workers, salt=b"\x07" * 32)

    path = out / "extract" / "2026" / "2026-10-27.parquet"
    assert path.exists()
    pf = pq.ParquetFile(path)
    assert pf.schema_arrow.remove_metadata().equals(schema.PARQUET_SCHEMA)
    assert pf.metadata.row_group(0).column(0).compression == "ZSTD"
    assert pf.schema_arrow.metadata[b"date"] == b"2026-10-27"
    assert not set(pf.schema_arrow.names) & schema.FORBIDDEN_COLUMNS

    table = pf.read()
    assert table.num_rows == entry["points_kept"] > 1000
    uids = table.column("uid").to_pylist()
    ts = table.column("t_utc").to_pylist()
    assert list(zip(uids, ts)) == sorted(zip(uids, ts))  # sorted by (uid, time)
    assert entry["aircraft"] == len(traces)
    assert entry["distinct_uids"] == len(set(uids))
    assert entry["status"] == "ok"

    # heatmap/ was skipped by range-seeking, not downloaded
    total = sum(s for _, s in parts)
    assert entry["bytes_fetched"] < total - 9 * 1024 * 1024

    # nothing from the fixtures' (already synthetic) addresses survives
    flat = json.dumps(table.to_pydict(), default=str)
    assert "f0000" not in flat


def test_same_aircraft_gets_different_uids_on_two_days(tmp_path, traces):
    parts = write_split_tar(tmp_path, traces)
    ex.extract_day(date(2026, 10, 27), tmp_path / "a", parts=parts, workers=1)
    ex.extract_day(date(2026, 10, 28), tmp_path / "a", parts=parts, workers=1)  # same input, new run, new salt
    d1 = pq.read_table(tmp_path / "a" / "extract" / "2026" / "2026-10-27.parquet").column("uid").to_pylist()
    d2 = pq.read_table(tmp_path / "a" / "extract" / "2026" / "2026-10-28.parquet").column("uid").to_pylist()
    assert len(d1) == len(d2) > 0
    assert not set(d1) & set(d2)


def test_non_trace_members_are_never_decoded(tmp_path, traces):
    parts = write_split_tar(tmp_path, traces[:3])
    f = SplitTarFile(parts)
    with tarfile.open(fileobj=f, mode="r:") as tar:
        names = [n for batch in ex.iter_trace_batches(tar, 2) for n in [len(batch)]]
    assert sum(names) == 3


def test_corrupt_trace_member_is_skipped(salt):
    table, stats = ex.process_batch([b"\x1f\x8bnot gzip", b"{broken"], salt)
    assert table is None and stats.traces == 0


def test_manifest_flags_a_day_far_below_the_trailing_median():
    days = {}
    for i in range(10):
        days[f"2026-10-{i + 1:02d}"] = {"points_kept": 1000}
    days["2026-10-11"] = {"points_kept": 700}   # 30% below the median: fine
    days["2026-10-12"] = {"points_kept": 590}   # 41% below: degraded
    days["2026-10-13"] = {"points_kept": 1000}
    ex.evaluate_manifest(days)
    assert days["2026-10-11"]["status"] == "ok"
    assert days["2026-10-12"]["status"] == "degraded"
    assert days["2026-10-13"]["status"] == "ok"
    assert all(days[f"2026-10-{i + 1:02d}"]["status"] == "ok" for i in range(7))  # not enough history to judge


def test_manifest_needs_history_before_judging():
    days = {f"2026-10-0{i}": {"points_kept": 1000} for i in range(1, 4)}
    days["2026-10-04"] = {"points_kept": 1}
    ex.evaluate_manifest(days)
    assert days["2026-10-04"]["status"] == "ok"


def test_update_manifest_round_trip(tmp_path):
    path = tmp_path / "days_manifest.json"
    ex.update_manifest(path, "2026-10-27", {"points_kept": 5})
    ex.update_manifest(path, "2026-10-28", {"points_kept": 6})
    data = json.loads(path.read_text())
    assert set(data["days"]) == {"2026-10-27", "2026-10-28"}
    assert data["schema_version"] == schema.SCHEMA_VERSION
