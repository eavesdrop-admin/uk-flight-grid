"""Split per-aircraft point streams into legs and resample them honestly.

Input is the time-ordered, airborne-only point stream for one pseudonymised
aircraft (Stage A output). Nothing here invents data: gaps longer than
MAX_INTERP_GAP_S are never bridged, they are recorded and reduce `continuity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

from geo import haversine_m

LEG_GAP_S = 15 * 60  # a silence this long starts a new leg
MIN_GS_KT = 40.0  # slower than this is a ground/taxi interval
MAX_INTERP_GAP_S = 120.0  # never interpolate across a longer gap
STEP_M = 100.0  # resample spacing


@dataclass(frozen=True)
class Point:
    t: float  # epoch seconds, UTC
    lat: float
    lon: float
    alt_ft: float | None  # AMSL; None means on the ground / unknown
    gs: float | None = None  # knots
    track: float | None = None  # degrees true
    baro_rate: float | None = None  # ft/min
    leg_flag: bool = False  # readsb "new leg" marker: this point starts a leg
    category: str | None = None  # ADS-B emitter category, A1..A7 etc.
    mil: bool = False
    heli: bool = False
    alt_src: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Point":
        """Build from a Stage A Parquet row (column names per the plan)."""
        t = row["t_utc"]
        if isinstance(t, datetime):
            t = t.timestamp()
        return cls(
            t=float(t),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            alt_ft=None if row.get("alt_ft") is None else float(row["alt_ft"]),
            gs=None if row.get("gs") is None else float(row["gs"]),
            track=None if row.get("track") is None else float(row["track"]),
            baro_rate=None if row.get("baro_rate") is None else float(row["baro_rate"]),
            leg_flag=bool(row.get("leg_flag") or False),
            category=row.get("category"),
            mil=bool(row.get("mil") or False),
            heli=bool(row.get("heli") or False),
            alt_src=row.get("alt_src"),
        )


@dataclass
class Leg:
    points: list[Point]

    @property
    def start_t(self) -> float:
        return self.points[0].t

    @property
    def end_t(self) -> float:
        return self.points[-1].t

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t

    @property
    def is_helicopter(self) -> bool:
        return any(p.heli or p.category == "A7" for p in self.points)

    @property
    def is_military(self) -> bool:
        return any(p.mil for p in self.points)


@dataclass(frozen=True)
class Gap:
    start_t: float
    end_t: float

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t


@dataclass(frozen=True)
class Sample:
    t: float
    lat: float
    lon: float
    alt_ft: float
    interpolated: bool


@dataclass
class Resampled:
    samples: list[Sample]
    gaps: list[Gap] = field(default_factory=list)  # unbridged gaps
    continuity: float = 1.0  # observed time / leg span


def split_legs(
    points: Iterable[Point],
    gap_s: float = LEG_GAP_S,
    min_gs_kt: float = MIN_GS_KT,
    min_points: int = 2,
) -> list[Leg]:
    """Split a point stream into legs.

    A new leg starts at a point flagged `leg_flag`, after a silence longer than
    gap_s, and after any ground / slower-than-min_gs_kt interval (those points
    are dropped). Duplicate timestamps keep the first point.
    """
    legs: list[Leg] = []
    cur: list[Point] = []
    last_t: float | None = None

    def flush() -> None:
        nonlocal cur
        if len(cur) >= min_points:
            legs.append(Leg(cur))
        cur = []

    for p in sorted(points, key=lambda p: p.t):
        if last_t is not None and p.t == last_t:
            continue
        on_ground = p.alt_ft is None or (p.gs is not None and p.gs < min_gs_kt)
        if on_ground:
            flush()
            last_t = p.t
            continue
        if cur and (p.leg_flag or p.t - cur[-1].t > gap_s):
            flush()
        cur.append(p)
        last_t = p.t
    flush()
    return legs


def resample(
    leg: Leg,
    step_m: float = STEP_M,
    max_gap_s: float = MAX_INTERP_GAP_S,
) -> Resampled:
    """Resample a leg to ~step_m spacing, bridging only gaps <= max_gap_s.

    Observed points are always emitted (interpolated=False). Between two
    consecutive points <= max_gap_s apart, intermediate samples are linearly
    interpolated every ~step_m. Longer gaps are recorded in `gaps` and nothing
    is emitted across them.
    """
    pts = leg.points
    samples = [Sample(pts[0].t, pts[0].lat, pts[0].lon, pts[0].alt_ft, False)]  # type: ignore[arg-type]
    gaps: list[Gap] = []
    gap_total = 0.0
    for a, b in zip(pts, pts[1:]):
        dt = b.t - a.t
        if dt > max_gap_s:
            gaps.append(Gap(a.t, b.t))
            gap_total += dt
        else:
            dist = haversine_m(a.lat, a.lon, b.lat, b.lon)
            n = int(math.ceil(dist / step_m))
            for i in range(1, n):
                f = i / n
                samples.append(
                    Sample(
                        t=a.t + f * dt,
                        lat=a.lat + f * (b.lat - a.lat),
                        lon=a.lon + f * (b.lon - a.lon),
                        alt_ft=a.alt_ft + f * (b.alt_ft - a.alt_ft),  # type: ignore[operator]
                        interpolated=True,
                    )
                )
        samples.append(Sample(b.t, b.lat, b.lon, b.alt_ft, False))  # type: ignore[arg-type]
    span = leg.duration_s
    continuity = 1.0 if span <= 0 else max(0.0, (span - gap_total) / span)
    return Resampled(samples=samples, gaps=gaps, continuity=continuity)
