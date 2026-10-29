from datetime import datetime, timedelta, timezone

import polars as pl
import pytest
from h3.api import basic_int as h3i

import coverage as cov
from grid_schema import COVERAGE_SCHEMA, STATUS_GOOD, STATUS_INSUFFICIENT, STATUS_PARTIAL

T0 = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
LAT, LON = 52.0, -1.0


def frame(rows):
    """rows: (uid, leg_flag, seconds, lat, lon, alt_ft)"""
    return pl.DataFrame(
        {
            "uid": [r[0] for r in rows],
            "leg_flag": [r[1] for r in rows],
            "t_utc": [T0 + timedelta(seconds=r[2]) for r in rows],
            "lat": [r[3] for r in rows],
            "lon": [r[4] for r in rows],
            "alt_ft": [r[5] for r in rows],
        }
    ).cast({"uid": pl.UInt64, "t_utc": pl.Datetime("ms", "UTC")})


def track(uid, n, *, dt=5, alt=800, lat=LAT, lon0=LON, dlon=0.001, start=0, first_flag=True):
    return [(uid, first_flag and i == 0, start + i * dt, lat, lon0 + i * dlon, alt) for i in range(n)]


# ------------------------------------------------------------ classify


@pytest.mark.parametrize(
    "floor,cont,pts,cross,expected",
    [
        (1500, 0.95, 500, 100, STATUS_GOOD),  # boundaries are inclusive
        (1501, 0.95, 500, 100, STATUS_PARTIAL),
        (4000, 0.95, 500, 100, STATUS_PARTIAL),
        (4001, 0.95, 500, 100, STATUS_INSUFFICIENT),
        (800, 0.85, 500, 100, STATUS_GOOD),
        (800, 0.849, 500, 100, STATUS_PARTIAL),
        (800, 0.60, 500, 100, STATUS_PARTIAL),
        (800, 0.599, 500, 100, STATUS_INSUFFICIENT),
        (800, 0.30, 500, 49, STATUS_GOOD),  # fewer than 50 crossings: continuity is not judged
        (800, 0.30, 500, 50, STATUS_INSUFFICIENT),
        (800, 0.95, 499, 100, STATUS_INSUFFICIENT),  # too few samples, however low the floor
        (3000, 0.95, 499, 100, STATUS_INSUFFICIENT),  # ... and never partial on a handful of points
        (3000, 0.30, 500, 100, STATUS_PARTIAL),  # floor 1,500-4,000: counts are a lower bound whatever the continuity
    ],
)
def test_classify_thresholds(floor, cont, pts, cross, expected):
    assert cov.classify(floor, cont, pts, cross) == expected


def test_classify_min_sample_points_is_adjustable():
    assert cov.classify(800, 0.95, 100, 100, min_sample_points=100) == STATUS_GOOD
    assert cov.classify(800, 0.95, 99, 100, min_sample_points=100) == STATUS_INSUFFICIENT


# ------------------------------------------------------- coverage_day


def stats_for(day, lat, lon):
    row = day.stats.filter(pl.col("cell") == h3i.latlng_to_cell(lat, lon, 7))
    return row.row(0, named=True) if row.height else None


def test_observed_time_and_points_are_counted_per_cell():
    day = cov.coverage_day(frame(track(1, 10)))
    assert day.stats["n_points"].sum() == 10
    assert day.stats["gap_s"].sum() == 0
    assert day.stats["obs_s"].sum() == pytest.approx(9 * 5)
    assert day.stats["crossings"].sum() >= 1
    assert day.hist["n"].sum() == 10


def test_gap_over_120s_between_low_points_is_lost_time_spread_over_cells_it_spans():
    # 6 km gap (about 1.2 cells wide) of 300 s between two low points of one leg
    rows = track(1, 5) + [(1, False, 20 + 300, LAT, LON + 0.004 + 0.09, 800)] + [(1, False, 325 + 5 * i, LAT, LON + 0.094 + 0.001 * i, 800) for i in range(4)]
    day = cov.coverage_day(frame(rows))
    assert day.stats["gap_s"].sum() == pytest.approx(300.0)
    # it was spread over more than one cell
    assert (day.stats["gap_s"] > 0).sum() >= 2


def test_gap_with_a_high_end_is_not_counted_as_a_receiver_hole():
    # cruise above 7,000 ft was dropped by Stage A: a long gap between a climb and a descent is not a hole
    rows = [(1, True, 0, LAT, LON, 6500), (1, False, 5, LAT, LON + 0.001, 9800),
            (1, False, 1500 - 900 + 5, LAT, LON + 0.6, 9500), (1, False, 1500 - 900 + 10, LAT, LON + 0.601, 6000)]
    day = cov.coverage_day(frame(rows))
    assert day.stats["gap_s"].sum() == 0


def test_silence_over_fifteen_minutes_is_a_new_leg_not_a_gap():
    rows = track(1, 3) + track(1, 3, start=3600, first_flag=False)
    day = cov.coverage_day(frame(rows))
    assert day.stats["gap_s"].sum() == 0


def test_leg_flag_and_uid_changes_break_pairs():
    rows = track(1, 3) + [(1, True, 100, LAT, LON, 800)] + track(2, 3, start=0)
    day = cov.coverage_day(frame(rows))
    # pairs: 2 in leg A, 0 across the flag, 2 in aircraft 2
    assert day.stats["obs_s"].sum() == pytest.approx(4 * 5)


# ------------------------------------------------------------ finalise


def dense_day(n, alt=800, uid=1, gap=False):
    rows = track(uid, n, dt=5, alt=alt, dlon=0.000001)
    return cov.coverage_day(frame(rows))


def test_good_cell_needs_enough_samples_and_a_low_floor():
    cd = dense_day(600)
    out = cov.finalise([cd])
    assert dict(out.schema) == COVERAGE_SCHEMA
    assert set(out["status"]) == {STATUS_GOOD}
    assert out["sample_points"].max() == 600 and out["floor_ft"].max() <= 900
    thin = cov.finalise([dense_day(100)])
    assert set(thin["status"]) == {STATUS_INSUFFICIENT}


def test_high_traffic_only_gives_partial_or_insufficient_never_good():
    mid = cov.finalise([dense_day(600, alt=3200)])
    assert set(mid["status"]) == {STATUS_PARTIAL}
    assert mid["floor_ft"].min() > 1500
    high = cov.finalise([dense_day(600, alt=8000)])
    assert set(high["status"]) == {STATUS_INSUFFICIENT}


def test_floor_is_the_fifth_percentile_not_the_minimum():
    # 1,000 points at 5,000 ft plus 20 stragglers (2%) at 300 ft: the 5th percentile ignores the stragglers
    rows = track(1, 1000, alt=5000, dlon=0.00005) + track(2, 20, alt=300, dlon=0.00005, start=0)
    out = cov.finalise([cov.coverage_day(frame(rows))])
    assert out["floor_ft"].min() >= 4900 and set(out["status"]) == {STATUS_INSUFFICIENT}


def test_rows_exist_only_for_cells_with_their_own_points_and_ring_pools_neighbours():
    centre = h3i.latlng_to_cell(LAT, LON, 7)
    neighbour = next(n for n in h3i.grid_disk(centre, 1) if n != centre)
    lat, lon = h3i.cell_to_latlng(neighbour)
    rows = track(1, 300, lat=LAT, lon0=LON, dlon=0.00001) + track(2, 300, lat=lat, lon0=lon, dlon=0.00001)
    out = cov.finalise([cov.coverage_day(frame(rows))])
    assert set(out["h3_r7"].to_list()) == {centre, neighbour}  # no rows for the empty ring cells
    assert out["sample_points"].to_list() == [600, 600]  # each pooled with the other


def test_days_are_pooled():
    out = cov.finalise([dense_day(300, uid=1), dense_day(300, uid=2)])
    assert out["sample_points"].max() == 600 and set(out["status"]) == {STATUS_GOOD}


def test_continuity_falls_with_gaps():
    rows = []
    t = 0
    for k in range(60):  # 60 hops of 10 s then a 200 s silence, repeated: a third of the time is lost
        rows += [(1, k == 0 and t == 0, t + 10 * i, LAT, LON + 0.0001 * (k * 12 + i), 800) for i in range(11)]
        t += 100 + 200
    out = cov.finalise([cov.coverage_day(frame(rows))])
    assert out["continuity"].min() < 0.85
    assert out["status"].to_list() != [STATUS_GOOD]


def test_downgrade_steps_one_level_and_only_for_named_cells():
    base = pl.DataFrame(
        {"h3_r7": [1, 2, 3, 4], "status": [STATUS_GOOD, STATUS_PARTIAL, STATUS_INSUFFICIENT, STATUS_GOOD],
         "floor_ft": [0] * 4, "continuity": [1.0] * 4, "sample_points": [1] * 4}
    )
    out = cov.downgrade(base, [1, 2, 3])
    assert out["status"].to_list() == [STATUS_PARTIAL, STATUS_INSUFFICIENT, STATUS_INSUFFICIENT, STATUS_GOOD]
