from pathlib import Path

import pytest

from attribute import attribute_aircraft, classify_leg, helicopter_movements
from reference import DEFAULT_DFT_PATH, build_reference
from segment import Leg, split_legs
from synth import final_approach, fly, offset

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ref():
    return build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", DEFAULT_DFT_PATH)


@pytest.fixture(scope="module")
def e(ref):
    return {r.ident: r for r in ref.airports["EGLL"].runways}


def leg_of(points):
    legs = split_legs(points)
    assert len(legs) == 1
    return legs[0]


def test_straight_final_to_27L_is_A_27L_not_27R(ref, e):
    a = classify_leg(leg_of(final_approach(e["27L"], 12, 300)), ref)
    assert (a.label, a.airport, a.runway, a.code) == ("A", "EGLL", "27L", "A 27L")
    assert not a.inferred


def test_straight_final_to_27R_is_27R(ref, e):
    a = classify_leg(leg_of(final_approach(e["27R"], 12, 300)), ref)
    assert a.code == "A 27R"


def test_final_with_small_lateral_error_keeps_nearest_centreline(ref, e):
    # 300 m right of the 27L centreline is still 27L (27R is 1.4 km away).
    assert classify_leg(leg_of(final_approach(e["27L"], 12, 300, lateral_m=300)), ref).code == "A 27L"
    # And 300 m left of 27R (towards 27L) is still 27R.
    assert classify_leg(leg_of(final_approach(e["27R"], 12, 300, lateral_m=300)), ref).code == "A 27R"


def test_easterly_final_to_09L(ref, e):
    assert classify_leg(leg_of(final_approach(e["09L"], 12, 300)), ref).code == "A 09L"


def test_final_midway_between_runways_gives_airport_but_unresolved_runway(ref, e):
    # ~706 m from each centreline: further than 600 m from both.
    a = classify_leg(leg_of(final_approach(e["27L"], 12, 300, lateral_m=-706)), ref)
    assert (a.label, a.airport, a.runway, a.code) == ("A", "EGLL", None, "A ?")


def test_departure_from_09R(ref, e):
    r = e["09R"]
    s1 = (*offset(r.lat, r.lon, 1500, 0), r.elevation_ft + 50)
    s2 = (*offset(r.lat, r.lon, 12000, 0), r.elevation_ft + 4000)
    a = classify_leg(leg_of(fly([s1, s2], gs_kt=160)), ref)
    assert (a.label, a.airport, a.runway, a.code) == ("D", "EGLL", "09R", "D 09R")


def test_departure_from_27L_and_not_09R(ref, e):
    r = e["27L"]
    s1 = (*offset(r.lat, r.lon, -1500, 0), r.elevation_ft + 50)
    s2 = (*offset(r.lat, r.lon, -12000, 0), r.elevation_ft + 4000)
    assert classify_leg(leg_of(fly([s1, s2], gs_kt=160)), ref).code == "D 27L"


def _circuit_27L(e):
    r = e["27L"]
    pts = [
        (-1500, 0, r.elevation_ft + 50),
        (-4500, 0, 1100),
        (-5500, -1500, 1100),  # crosswind, left turn
        (3000, -1500, 1100),  # downwind
        (3000, 0, 700),  # base
        (400, 0, r.elevation_ft + 150),  # final
    ]
    return [(*offset(r.lat, r.lon, x, y), alt) for x, y, alt in pts]


def test_circuit_in_one_leg(ref, e):
    a = classify_leg(leg_of(fly(_circuit_27L(e), gs_kt=100)), ref)
    assert a.label == "C" and a.airport == "EGLL" and a.runway == "27L"


def test_circuit_across_two_legs_pairs_arrival_and_departure(ref, e):
    r27l = e["27L"]
    arr = final_approach(r27l, 12, 300, t0=1_700_000_000.0)
    dep_start = arr[-1].t + 5 * 60  # lands, goes round again 5 minutes later
    s1 = (*offset(r27l.lat, r27l.lon, -1500, 0), r27l.elevation_ft + 50)
    s2 = (*offset(r27l.lat, r27l.lon, -12000, 0), r27l.elevation_ft + 4000)
    dep = fly([s1, s2], gs_kt=160, t0=dep_start)
    legs = split_legs(arr + dep)
    assert len(legs) == 2
    assert [a.label for a in [classify_leg(l, ref) for l in legs]] == ["A", "D"]
    out = attribute_aircraft(legs, ref)
    assert [a.label for a in out] == ["C", "C"]
    assert {a.airport for a in out} == {"EGLL"}


def test_arrival_then_departure_far_apart_stay_A_and_D(ref, e):
    r = e["27L"]
    arr = final_approach(r, 12, 300, t0=1_700_000_000.0)
    s1 = (*offset(r.lat, r.lon, -1500, 0), r.elevation_ft + 50)
    s2 = (*offset(r.lat, r.lon, -12000, 0), r.elevation_ft + 4000)
    dep = fly([s1, s2], gs_kt=160, t0=arr[-1].t + 3 * 3600)
    assert [a.label for a in attribute_aircraft(split_legs(arr + dep), ref)] == ["A", "D"]


def test_high_level_pass_near_heathrow_is_overflight(ref, e):
    r = e["27L"]
    a, b = offset(r.lat, r.lon, -30000, 2000), offset(r.lat, r.lon, 30000, 2000)
    out = classify_leg(leg_of(fly([(*a, 4000), (*b, 4000)], gs_kt=250)), ref)
    assert (out.label, out.airport, out.runway, out.code) == ("O", None, None, "O")


def test_remote_low_track_is_overflight(ref):
    out = classify_leg(leg_of(fly([(53.0, -1.0, 1500), (53.0, -0.5, 1500)], gs_kt=120)), ref)
    assert out.label == "O"


def test_level_low_pass_down_the_centreline_is_not_an_arrival(ref, e):
    r = e["27L"]
    a, b = offset(r.thr_lat, r.thr_lon, 12000, 0), offset(r.thr_lat, r.thr_lon, 400, 0)
    out = classify_leg(leg_of(fly([(*a, 1500), (*b, 1500)], gs_kt=140)), ref)
    assert out.label == "O"


def test_descending_but_far_above_glide_is_not_inferred_arrival(ref, e):
    r = e["27L"]
    a, b = offset(r.thr_lat, r.thr_lon, 25000, 0), offset(r.thr_lat, r.thr_lon, 12000, 0)
    out = classify_leg(leg_of(fly([(*a, 6000), (*b, 5000)], gs_kt=200)), ref)
    assert out.label == "O"


def test_trace_lost_ten_km_out_is_completed_and_flagged_inferred(ref, e):
    a = classify_leg(leg_of(final_approach(e["27L"], 14, 10_000)), ref)
    assert a.code == "A 27L" and a.inferred
    tail = a.inferred_tail
    assert tail and all(s.interpolated for s in tail)
    r = e["27L"]
    assert (tail[-1].lat, tail[-1].lon) == (r.thr_lat, r.thr_lon)
    assert [s.alt_ft for s in tail] == sorted((s.alt_ft for s in tail), reverse=True)
    # Within 8 km nothing is synthesised.
    assert not classify_leg(leg_of(final_approach(e["27L"], 14, 300)), ref).inferred_tail


def test_trace_lost_beyond_synthesis_radius_is_not_an_arrival(ref, e):
    assert classify_leg(leg_of(final_approach(e["27L"], 25, 18_000)), ref).label == "O"


def test_helicopter_by_category_and_by_flag_wins_over_arrival(ref, e):
    track = final_approach(e["27L"], 12, 300, category="A7")
    assert classify_leg(leg_of(track), ref).label == "H"
    assert classify_leg(leg_of(final_approach(e["27L"], 12, 300, heli=True)), ref).label == "H"


def test_helicopter_landing_and_lifting_at_an_airport_count_as_movements(ref):
    a = ref.airports["EGLL"]
    far = offset(a.lat, a.lon, 0, 20_000)
    land = leg_of(fly([(*far, 2_000), (a.lat, a.lon, a.elevation_ft + 100)], gs_kt=100, category="A7"))
    assert helicopter_movements(land, ref) == [("EGLL", "A")]
    lift = leg_of(fly([(a.lat, a.lon, a.elevation_ft + 100), (*far, 2_000)], gs_kt=100, category="A7"))
    assert helicopter_movements(lift, ref) == [("EGLL", "D")]


def test_helicopter_passing_high_or_far_is_not_a_movement(ref):
    a = ref.airports["EGLL"]
    far, other = offset(a.lat, a.lon, 0, 20_000), offset(a.lat, a.lon, 0, -20_000)
    assert helicopter_movements(leg_of(fly([(*far, 1_500), (a.lat, a.lon, 1_500), (*other, 1_500)], gs_kt=100, category="A7")), ref) == []
    assert helicopter_movements(leg_of(fly([(*far, 2_000), (*offset(a.lat, a.lon, 0, 3_000), a.elevation_ft + 100)], gs_kt=100, category="A7")), ref) == []


def test_military_wins_over_arrival(ref, e):
    a = classify_leg(leg_of(final_approach(e["27L"], 12, 300, mil=True)), ref)
    assert (a.label, a.airport, a.runway) == ("M", None, None)


def test_military_helicopter_is_H(ref, e):
    assert classify_leg(leg_of(final_approach(e["27L"], 12, 300, mil=True, heli=True)), ref).label == "H"


def test_other_airports_resolve_their_own_runways(ref):
    egcc = {r.ident: r for r in ref.airports["EGCC"].runways}
    assert classify_leg(leg_of(final_approach(egcc["23R"], 12, 300)), ref).code == "A 23R"
    assert classify_leg(leg_of(final_approach(egcc["05L"], 12, 300)), ref).code == "A 05L"
    egff = {r.ident: r for r in ref.airports["EGFF"].runways}
    assert classify_leg(leg_of(final_approach(egff["12"], 12, 300)), ref).code == "A 12"


def test_gap_inside_final_does_not_break_attribution(ref, e):
    # A 3-minute hole in the middle of a long straight-in final: legs stay one leg
    # (under 15 min), and the arrival is still read from the last minutes.
    pts = final_approach(e["27L"], 25, 300)
    mid = len(pts) // 2
    from dataclasses import replace

    shifted = pts[:mid] + [replace(p, t=p.t + 180) for p in pts[mid:]]
    out = classify_leg(Leg(shifted), ref)
    assert out.code == "A 27L"
