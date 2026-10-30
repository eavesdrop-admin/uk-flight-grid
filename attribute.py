"""Classify legs: arrival / departure to a runway end, circuit, overflight,
helicopter, military.

Labels (plan section 5):
    H helicopter, M military, C circuit, A arrival, D departure, O overflight
Precedence H > M > C > A > D > O. An arrival or departure whose airport is
certain but whose runway end cannot be resolved is labelled with runway None
(printed `A ?`), never guessed.

All altitudes are AMSL feet; runway elevation comes from OurAirports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from geo import FT_PER_M, along_cross_m, angdiff, bearing_deg, circular_mean_deg, destination, haversine_m
from reference import Reference, RunwayEnd
from segment import Leg, Point, Sample

# Arrival
ARR_RADIUS_M = 8_000.0  # last airborne point this close to a landing threshold
ARR_MAX_AGL_FT = 2_000.0
ARR_MAX_RATE_FPM = -300.0  # mean vertical rate over the last minute
# Departure
DEP_RADIUS_M = 4_000.0  # first airborne point this close to a runway
DEP_MAX_AGL_FT = 2_000.0
DEP_MIN_RATE_FPM = 300.0
# Runway-end resolution
ALIGN_DEG = 15.0
CROSS_TRACK_M = 600.0
COURSE_WINDOW_S = 120.0  # arrival: mean track over the last 2 min (plan)
DEP_COURSE_WINDOW_S = 45.0  # departure: initial climb-out only, so an early crosswind turn does not hide the runway
RATE_WINDOW_S = 60.0
# Final-approach completion for traces lost before the threshold
SYNTH_RADIUS_M = 15_000.0
GLIDE_DEG = 3.0
GLIDE_TOL_FT = 800.0
TCH_FT = 50.0
SYNTH_STEP_M = 100.0
# Circuit: arrival and departure at one airport within this long
CIRCUIT_WINDOW_S = 20 * 60.0
# Helicopter movement, for the capture ratio only: trace starts or ends low, near the airport.
# DfT counts helicopter air transport movements (Norwich offshore flights are most of its total),
# so leaving them out made the ratio say Norwich was under-captured when it was not.
HELI_RADIUS_M = 1_500.0
HELI_MAX_AGL_FT = 500.0


@dataclass(frozen=True)
class Movement:
    airport: str
    runway: str | None  # None = airport certain, runway end not resolved
    inferred: bool = False  # final approach completed by synthesis


@dataclass
class Attribution:
    label: str  # one of A D C O H M
    airport: str | None = None
    runway: str | None = None
    arrival: Movement | None = None
    departure: Movement | None = None
    inferred_tail: list[Sample] = field(default_factory=list)

    @property
    def inferred(self) -> bool:
        return bool(self.arrival and self.arrival.inferred)

    @property
    def code(self) -> str:
        """`A 27L`, `D 09R`, `A ?` (runway unresolved), `O`, ..."""
        if self.label in ("A", "D", "C"):
            return f"{self.label} {self.runway or '?'}"
        return self.label


# ---------------------------------------------------------------- helpers


def _window(points: Sequence[Point], seconds: float, from_end: bool) -> list[Point]:
    if from_end:
        t0 = points[-1].t - seconds
        return [p for p in points if p.t >= t0]
    t1 = points[0].t + seconds
    return [p for p in points if p.t <= t1]


def _vertical_rate(win: Sequence[Point]) -> float | None:
    """Mean vertical rate in ft/min: reported rates, else from altitude change."""
    rates = [p.baro_rate for p in win if p.baro_rate is not None]
    if rates:
        return sum(rates) / len(rates)
    if len(win) >= 2 and win[-1].t > win[0].t and win[0].alt_ft is not None and win[-1].alt_ft is not None:
        return (win[-1].alt_ft - win[0].alt_ft) / ((win[-1].t - win[0].t) / 60.0)
    return None


def _course(win: Sequence[Point]) -> float | None:
    tracks = [p.track for p in win if p.track is not None]
    if tracks:
        return circular_mean_deg(tracks)
    if len(win) >= 2:
        a, b = win[0], win[-1]
        if (a.lat, a.lon) != (b.lat, b.lon):
            return bearing_deg(a.lat, a.lon, b.lat, b.lon)
    return None


def _segment_distance_m(r: RunwayEnd, lat: float, lon: float) -> float:
    """Distance from a point to the runway centreline segment of end r."""
    along, cross = along_cross_m(r.lat, r.lon, r.heading, lat, lon)
    if along < 0:
        return math.hypot(along, cross)
    if along > r.length_m:
        return math.hypot(along - r.length_m, cross)
    return abs(cross)


def _glide_height_ft(dist_m: float) -> float:
    return TCH_FT + dist_m * math.tan(math.radians(GLIDE_DEG)) * FT_PER_M


def _synth_tail(r: RunwayEnd, from_dist_m: float) -> list[Sample]:
    """Points down the extended centreline at a 3 degree path, to the threshold."""
    back = (r.heading + 180.0) % 360.0
    out = []
    d = from_dist_m - SYNTH_STEP_M
    while d > 0:
        lat, lon = destination(r.thr_lat, r.thr_lon, back, d)
        out.append(Sample(t=float("nan"), lat=lat, lon=lon, alt_ft=r.elevation_ft + _glide_height_ft(d), interpolated=True))
        d -= SYNTH_STEP_M
    out.append(Sample(t=float("nan"), lat=r.thr_lat, lon=r.thr_lon, alt_ft=r.elevation_ft + TCH_FT, interpolated=True))
    return out


# -------------------------------------------------------------- detection


def detect_arrival(leg: Leg, ref: Reference) -> tuple[Movement, list[Sample]] | None:
    pts = leg.points
    last = pts[-1]
    if last.alt_ft is None:
        return None
    near = ref.ends_near(last.lat, last.lon, SYNTH_RADIUS_M)
    if not near:
        return None

    rate = _vertical_rate(_window(pts, RATE_WINDOW_S, from_end=True))
    if rate is None or rate >= ARR_MAX_RATE_FPM:
        return None
    course = _course(_window(pts, COURSE_WINDOW_S, from_end=True))

    best: tuple[float, RunwayEnd, float, bool] | None = None  # |cross|, end, dist, inferred
    nearest: tuple[float, RunwayEnd] | None = None  # for the airport-only fallback
    for r in near:
        dist = haversine_m(last.lat, last.lon, r.thr_lat, r.thr_lon)
        agl = last.alt_ft - r.elevation_ft
        if dist <= ARR_RADIUS_M and agl <= ARR_MAX_AGL_FT and (nearest is None or dist < nearest[0]):
            nearest = (dist, r)
        if course is None or abs(angdiff(course, r.heading)) > ALIGN_DEG:
            continue
        along, cross = along_cross_m(r.thr_lat, r.thr_lon, r.heading, last.lat, last.lon)
        if abs(cross) > CROSS_TRACK_M or along > r.length_m:
            continue  # off the centreline, or beyond the far end of the runway
        if dist <= ARR_RADIUS_M and agl <= ARR_MAX_AGL_FT and along <= r.length_m:
            cand = (abs(cross), r, dist, False)
        elif along < 0 and dist <= SYNTH_RADIUS_M and agl <= _glide_height_ft(dist) + GLIDE_TOL_FT:
            cand = (abs(cross), r, dist, True)
        else:
            continue
        if best is None or cand[0] < best[0]:
            best = cand

    if best is not None:
        _, r, dist, inferred = best
        return Movement(r.airport, r.ident, inferred), (_synth_tail(r, dist) if inferred else [])
    if nearest is not None:
        return Movement(nearest[1].airport, None), []
    return None


def detect_departure(leg: Leg, ref: Reference) -> Movement | None:
    pts = leg.points
    first = pts[0]
    if first.alt_ft is None:
        return None
    near = ref.starts_near(first.lat, first.lon, DEP_RADIUS_M + 6_000.0)  # runways are up to ~5 km
    if not near:
        return None

    rate = _vertical_rate(_window(pts, RATE_WINDOW_S, from_end=False))
    if rate is None or rate <= DEP_MIN_RATE_FPM:
        return None
    course = _course(_window(pts, DEP_COURSE_WINDOW_S, from_end=False))

    best: tuple[float, RunwayEnd] | None = None
    nearest: tuple[float, RunwayEnd] | None = None
    for r in near:
        if first.alt_ft - r.elevation_ft > DEP_MAX_AGL_FT:
            continue
        seg = _segment_distance_m(r, first.lat, first.lon)
        if seg > DEP_RADIUS_M:
            continue
        if nearest is None or seg < nearest[0]:
            nearest = (seg, r)
        if course is None or abs(angdiff(course, r.heading)) > ALIGN_DEG:
            continue
        along, cross = along_cross_m(r.lat, r.lon, r.heading, first.lat, first.lon)
        if abs(cross) > CROSS_TRACK_M or along < -500.0:
            continue  # behind the start of this runway end
        if best is None or abs(cross) < best[0]:
            best = (abs(cross), r)

    if best is not None:
        return Movement(best[1].airport, best[1].ident)
    if nearest is not None:
        return Movement(nearest[1].airport, None)
    return None


def _heli_airport(p: Point, ref: Reference) -> str | None:
    """Airport whose reference point is within HELI_RADIUS_M of p, with p below HELI_MAX_AGL_FT."""
    best = None
    for a in ref.airports.values():
        if abs(a.lat - p.lat) > HELI_RADIUS_M / 111_000.0 + 0.01:
            continue
        d = haversine_m(p.lat, p.lon, a.lat, a.lon)
        if d <= HELI_RADIUS_M and p.alt_ft - a.elevation_ft <= HELI_MAX_AGL_FT and (best is None or d < best[0]):
            best = (d, a.icao)
    return best[1] if best else None


def helicopter_movements(leg: Leg, ref: Reference) -> list[tuple[str, str]]:
    """(airport, 'A' or 'D') for a helicopter leg that ends or starts low at an airport."""
    out = []
    dep = _heli_airport(leg.points[0], ref)
    if dep:
        out.append((dep, "D"))
    arr = _heli_airport(leg.points[-1], ref)
    if arr:
        out.append((arr, "A"))
    return out


# --------------------------------------------------------- classification


def classify_leg(leg: Leg, ref: Reference) -> Attribution:
    """Label one leg. Cross-leg circuits are resolved by attribute_aircraft()."""
    if leg.is_helicopter:
        return Attribution("H")
    if leg.is_military:
        return Attribution("M")

    arr = detect_arrival(leg, ref)
    dep = detect_departure(leg, ref)
    arrival, tail = (arr if arr else (None, []))

    if arrival and dep and arrival.airport == dep.airport and leg.duration_s <= CIRCUIT_WINDOW_S:
        return Attribution("C", arrival.airport, arrival.runway or dep.runway, arrival, dep, tail)
    if arrival:
        return Attribution("A", arrival.airport, arrival.runway, arrival, None, tail)
    if dep:
        return Attribution("D", dep.airport, dep.runway, None, dep)
    return Attribution("O")


def attribute_aircraft(legs: Sequence[Leg], ref: Reference) -> list[Attribution]:
    """Label every leg of one aircraft, then pair arrival/departure legs at one
    airport within CIRCUIT_WINDOW_S as circuits (precedence C > A > D)."""
    out = [classify_leg(leg, ref) for leg in legs]
    for i in range(len(legs) - 1):
        a, b = out[i], out[i + 1]
        if a.label == "A" and b.label == "D":
            gap_s = legs[i + 1].start_t - legs[i].end_t
        elif a.label == "D" and b.label == "A":
            gap_s = legs[i + 1].end_t - legs[i].start_t
        else:
            continue
        if a.airport != b.airport or gap_s > CIRCUIT_WINDOW_S:
            continue
        arr_att, dep_att = (a, b) if a.label == "A" else (b, a)
        runway = arr_att.runway or dep_att.runway
        for x in (a, b):
            x.label = "C"
            x.runway = runway
            x.arrival = arr_att.arrival
            x.departure = dep_att.departure
    return out
