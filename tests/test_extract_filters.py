from __future__ import annotations

import pytest

import extract_day as ex
from schema import GEOID_OFFSET_FT
from trace_builders import DAY0, pt, trace

STEP = 0.001  # degrees of latitude, about 111 m


def run(points, salt, **extra):
    cols, stats = ex.extract_trace(trace(points, **extra), salt)
    return cols, stats


def test_keeps_a_plain_airborne_uk_point(salt):
    cols, stats = run([pt(0)], salt)
    assert cols["alt_ft"] == [3000]
    assert cols["alt_src"] == ["baro"]
    assert cols["t_utc"] == [round(DAY0 * 1000)]
    assert cols["leg_flag"] == [True]  # first kept point of an aircraft starts a leg
    assert (stats.traces, stats.aircraft_uk, stats.points_in_bbox, stats.points_kept) == (1, 1, 1, 1)


@pytest.mark.parametrize(
    "point",
    [
        pt(0, lat=48.85, lon=2.35),                 # Paris: outside the box
        pt(0, lat=51.5, lon=-12.0),                 # Atlantic
        pt(0, alt="ground"),                        # on the ground
        pt(0, gs=39.0),                             # too slow
        pt(0, gs=None),                             # no ground speed
        pt(0, flags=1),                             # stale position
        pt(0, alt=10_001),                          # above ceiling
        pt(0, alt=None, geom=None),                 # no altitude at all
    ],
)
def test_point_is_dropped(point, salt):
    cols, stats = run([point], salt)
    assert cols is None
    assert stats.points_kept == 0
    assert stats.points_total == 1


def test_altitude_ceiling_is_inclusive_and_boundaries_kept(salt):
    cols, _ = run([pt(0, alt=10_000), pt(60, alt=10_001)], salt)
    assert cols["alt_ft"] == [10_000]
    cols, _ = run([pt(0, gs=40.0)], salt)
    assert cols is not None  # 40 kt exactly is kept


def test_corrupt_low_altitudes_are_dropped(salt):
    cols, _ = run([pt(0, alt=-800), pt(60, alt=-1001), pt(120, alt=3000, geom=-3000)], salt)
    assert cols["alt_ft"] == [-800]


def test_geometric_altitude_preferred_with_geoid_correction(salt):
    cols, _ = run([pt(0, alt=3000, geom=3165 + 100)], salt)
    assert cols["alt_ft"] == [round(3265 - GEOID_OFFSET_FT)]
    assert cols["alt_src"] == ["geom"]


def test_baro_used_when_geometric_missing(salt):
    cols, _ = run([pt(0, alt=2500, geom=None)], salt)
    assert cols["alt_ft"] == [2500]
    assert cols["alt_src"] == ["baro"]


def test_flag_bit3_means_altitude_field_is_geometric(salt):
    cols, _ = run([pt(0, alt=2165, flags=8)], salt)
    assert cols["alt_ft"] == [2000]
    assert cols["alt_src"] == ["geom"]


def test_ceiling_applies_after_geoid_correction(salt):
    cols, _ = run([pt(0, alt=9000, geom=10_164)], salt)  # 9,999 ft AMSL
    assert cols["alt_ft"] == [9999]
    cols, _ = run([pt(0, alt=9000, geom=10_166)], salt)  # 10,001 ft AMSL
    assert cols is None


def test_thinning_keeps_8s_or_400m(salt):
    pts = [pt(i, lat=51.5 + i * STEP) for i in range(6)]  # 1 s apart, 111 m apart
    pts.append(pt(12, lat=51.5 + 5 * STEP))               # 8 s after the last kept, not moved
    pts.append(pt(13, lat=51.5 + 5 * STEP))               # 1 s later, not moved: dropped
    cols, _ = run(pts, salt)
    kept_offsets = [round(t / 1000 - DAY0) for t in cols["t_utc"]]
    assert kept_offsets == [0, 4, 12]


def test_thinning_never_drops_the_first_point_after_a_break(salt):
    pts = [pt(0), pt(1, flags=2)]  # new leg one second later, same place
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, True]
    assert len(cols["t_utc"]) == 2


def test_new_leg_flag_on_a_dropped_point_carries_to_next_kept_point(salt):
    pts = [pt(0), pt(60, alt=20_000, flags=2), pt(120, alt=4000)]
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, True]


def test_ground_interval_marks_next_point_as_new_leg(salt):
    pts = [
        pt(0, alt=3000),
        pt(60, alt=2000),
        pt(120, alt="ground", gs=20.0),
        pt(180, alt="ground", gs=0.0),
        pt(600, alt=1500),
        pt(660, alt=1800),
    ]
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, False, True, False]


def test_slow_interval_marks_new_leg(salt):
    pts = [pt(0), pt(60), pt(120, gs=10.0), pt(180)]
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, False, True]


def test_new_leg_flag_outside_the_box_is_honoured(salt):
    pts = [pt(0), pt(60, lat=48.0, flags=2), pt(7200, lat=51.5)]
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, True]


def test_stale_points_do_not_break_legs(salt):
    pts = [pt(0), pt(30, flags=1), pt(60)]
    cols, _ = run(pts, salt)
    assert cols["leg_flag"] == [True, False]


def test_stats_count_all_and_in_box_points(salt):
    pts = [pt(0), pt(10, lat=48.0), pt(20, alt="ground"), pt(30, alt=20_000)]
    _, stats = run(pts, salt)
    assert stats.points_total == 4
    assert stats.points_in_bbox == 3
    assert stats.points_kept == 1


def test_aircraft_never_in_box_is_not_counted_as_uk(salt):
    _, stats = run([pt(0, lat=40.0, lon=-70.0)], salt)
    assert (stats.traces, stats.aircraft_uk) == (1, 0)


def test_category_is_taken_from_any_point_and_applied_to_all(salt):
    pts = [pt(0), pt(60, obj={"category": "A7", "flight": "SECRET1", "squawk": "7000"}), pt(120)]
    cols, _ = run(pts, salt)
    assert cols["category"] == ["A7"] * 3
    assert cols["heli"] == [True] * 3


def test_fixed_wing_is_not_heli_and_missing_category_is_null(salt):
    cols, _ = run([pt(0, obj={"category": "A3"}), pt(60)], salt)
    assert cols["heli"] == [False, False]
    cols, _ = run([pt(0)], salt)
    assert cols["category"] == [None]


def test_military_flag_is_bit_zero_of_db_flags(salt):
    assert run([pt(0)], salt, dbFlags=1)[0]["mil"] == [True]
    assert run([pt(0)], salt, dbFlags=8)[0]["mil"] == [False]  # LADD is not military
    assert run([pt(0)], salt, dbFlags=9)[0]["mil"] == [True]
    assert run([pt(0)], salt)[0]["mil"] == [False]
    assert run([pt(0)], salt, dbFlags=None)[0]["mil"] == [False]


def test_vertical_rate_falls_back_to_geometric(salt):
    cols, _ = run([pt(0, rate=-700), pt(60, rate=None, geom_rate=-650), pt(120, rate=None)], salt)
    assert cols["baro_rate"] == [-700, -650, None]


def test_mlat_points_are_counted_but_kept(salt):
    _, stats = run([pt(0, src="mlat"), pt(60)], salt)
    assert stats.points_kept == 2
    assert stats.points_kept_mlat == 1


def test_short_or_malformed_points_are_skipped(salt):
    cols, _ = run([[0, 51.5, -0.5], pt(60)], salt)
    assert cols["t_utc"] == [round((DAY0 + 60) * 1000)]


def test_identifying_fields_never_reach_the_output(salt):
    obj = trace(
        [pt(0, obj={"category": "A3", "flight": "BAW123  ", "squawk": "1234"})],
        icao="4ca7b2",
        r="G-ABCD",
        ownOp="Some Operator Ltd",
        desc="BOEING 737-800",
        year="2012",
        t="B738",
    )
    cols, _ = ex.extract_trace(obj, salt)
    flat = repr(cols)
    for secret in ("G-ABCD", "Some Operator", "BOEING", "2012", "B738", "BAW123", "1234", "4ca7b2"):
        assert secret not in flat


def test_missing_timestamp_or_icao_yields_nothing(salt):
    assert ex.extract_trace({"icao": "a00001", "trace": [pt(0)]}, salt)[0] is None
    assert ex.extract_trace({"timestamp": DAY0, "trace": [pt(0)]}, salt)[0] is None
