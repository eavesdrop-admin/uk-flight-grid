import pytest

from segment import MAX_INTERP_GAP_S, Point, resample, split_legs
from synth import fly, offset

LAT, LON = 52.0, -1.0


def _line(t0=0.0, n=10, dt=10.0, **kw):
    return [Point(t=t0 + i * dt, lat=LAT, lon=LON + i * 0.001, alt_ft=3000, gs=200, track=90, **kw) for i in range(n)]


def test_one_continuous_leg():
    legs = split_legs(_line())
    assert len(legs) == 1 and len(legs[0].points) == 10


def test_splits_on_leg_flag_long_gap_and_slow_interval():
    a = _line(0)
    b = [Point(t=p.t + 100, lat=p.lat, lon=p.lon + 0.1, alt_ft=3000, gs=200, track=90, leg_flag=(i == 0)) for i, p in enumerate(_line(0))]
    assert len(split_legs(a + b)) == 2  # leg_flag

    c = _line(0) + _line(2000)  # 2,000 s > 15 min of silence
    assert len(split_legs(c)) == 2

    slow = Point(t=95, lat=LAT, lon=LON, alt_ft=3000, gs=10, track=90)
    assert len(split_legs(_line(0, 10) + [slow] + _line(100, 5))) == 2  # slow point is dropped, splits

    assert len(split_legs(_line(0) + _line(100, leg_flag=False))) == 1


def test_short_fragments_and_duplicate_timestamps():
    one = [Point(t=0, lat=LAT, lon=LON, alt_ft=3000, gs=200)]
    assert split_legs(one) == []
    dup = _line(0, 3) + _line(0, 3)
    assert len(split_legs(dup)[0].points) == 3


def test_interpolates_across_short_gap_at_100m_steps():
    pts = [Point(t=0, lat=LAT, lon=LON, alt_ft=1000, gs=200), Point(t=60, lat=LAT, lon=LON + 0.05, alt_ft=2000, gs=200)]
    rs = resample(split_legs(pts)[0])
    assert rs.gaps == [] and rs.continuity == 1.0
    inter = [s for s in rs.samples if s.interpolated]
    assert len(inter) > 30  # ~3.4 km at 100 m
    assert all(1000 < s.alt_ft < 2000 for s in inter)
    assert [s.t for s in rs.samples] == sorted(s.t for s in rs.samples)


def test_three_minute_gap_is_not_interpolated():
    # 240 kt is ~123 m/s: the 180 s hole spans ~22 km.
    before = fly([(LAT, LON, 3000), (LAT, LON + 0.05, 3000)], gs_kt=240, dt=10, leg_flag=True)
    after = fly([(LAT, LON + 0.4, 3000), (LAT, LON + 0.45, 3000)], gs_kt=240, dt=10, leg_flag=False,
                t0=before[-1].t + 180)
    legs = split_legs(before + after)
    assert len(legs) == 1  # 180 s is well under the 15 min leg-split gap
    rs = resample(legs[0])
    assert len(rs.gaps) == 1 and rs.gaps[0].duration_s == 180 > MAX_INTERP_GAP_S
    g = rs.gaps[0]
    assert not any(g.start_t < s.t < g.end_t for s in rs.samples)  # nothing invented inside the hole
    assert any(s.interpolated for s in rs.samples)  # but real stretches are still densified
    assert rs.continuity == pytest.approx(1 - 180 / legs[0].duration_s)


def test_gap_exactly_at_threshold_is_still_bridged():
    pts = [Point(t=0, lat=LAT, lon=LON, alt_ft=1000, gs=200), Point(t=MAX_INTERP_GAP_S, lat=LAT, lon=LON + 0.02, alt_ft=1000, gs=200)]
    rs = resample(split_legs(pts)[0])
    assert rs.gaps == [] and any(s.interpolated for s in rs.samples)


def test_from_row_accepts_parquet_style_rows():
    from datetime import datetime, timezone

    p = Point.from_row({
        "t_utc": datetime(2026, 10, 27, 12, 0, 5, tzinfo=timezone.utc), "lat": 51.5, "lon": -0.4, "alt_ft": 900,
        "gs": 140.0, "track": None, "baro_rate": -700, "leg_flag": True, "category": "A3", "mil": False, "heli": False,
        "alt_src": "geom",
    })
    assert p.t == datetime(2026, 10, 27, 12, 0, 5, tzinfo=timezone.utc).timestamp() and p.track is None and p.leg_flag and p.category == "A3"
