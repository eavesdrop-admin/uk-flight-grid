"""Small spherical-geometry helpers (pure Python, no dependencies).

Distances here are at most a few tens of km, so a local tangent plane is
accurate to well under a metre per km. That is far below the 600 m
centreline tolerance used by attribute.py.
"""

from __future__ import annotations

import math

EARTH_R_M = 6_371_008.8
FT_PER_M = 3.280839895
M_PER_FT = 0.3048


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing from point 1 to point 2, degrees in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    y = math.sin(dlmb) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    """Point reached travelling dist_m along a great circle from (lat, lon)."""
    d = dist_m / EARTH_R_M
    b = math.radians(bearing)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def angdiff(a: float, b: float) -> float:
    """Signed smallest difference a - b in degrees, in (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def circular_mean_deg(angles: list[float]) -> float | None:
    if not angles:
        return None
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return None
    return math.degrees(math.atan2(s, c)) % 360.0


def along_cross_m(
    ref_lat: float, ref_lon: float, heading: float, lat: float, lon: float
) -> tuple[float, float]:
    """Position of (lat, lon) in a frame anchored at ref, x-axis along `heading`.

    Returns (along, cross) in metres. along > 0 is ahead of ref in the direction
    of travel; cross > 0 is to the right of the heading.
    """
    east = math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat)) * EARTH_R_M
    north = math.radians(lat - ref_lat) * EARTH_R_M
    h = math.radians(heading)
    along = east * math.sin(h) + north * math.cos(h)
    cross = east * math.cos(h) - north * math.sin(h)
    return along, cross
