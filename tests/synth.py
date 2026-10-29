"""Synthetic-track builders for attribution tests (pure Python, no network)."""

from __future__ import annotations

import math

from geo import bearing_deg, destination, haversine_m
from segment import Point

KT_TO_MS = 0.514444


def offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a point by east/north metres (flat-earth, fine for a few km)."""
    d = math.hypot(east_m, north_m)
    if d == 0:
        return lat, lon
    return destination(lat, lon, math.degrees(math.atan2(east_m, north_m)) % 360, d)


def fly(
    waypoints: list[tuple[float, float, float]],
    gs_kt: float = 140.0,
    t0: float = 1_700_000_000.0,
    dt: float = 5.0,
    leg_flag: bool = True,
    **flags,
) -> list[Point]:
    """Fly a polyline of (lat, lon, alt_ft) at constant ground speed.

    Altitude is linear in distance along the path; track and vertical rate are
    derived from the path, as a receiver would report them.
    """
    cum = [0.0]
    for a, b in zip(waypoints, waypoints[1:]):
        cum.append(cum[-1] + haversine_m(a[0], a[1], b[0], b[1]))
    step = gs_kt * KT_TO_MS * dt
    dists = [i * step for i in range(int(cum[-1] // step) + 1)]
    if cum[-1] - dists[-1] > 1.0:
        dists.append(cum[-1])

    def at(d: float):
        i = max(j for j in range(len(cum) - 1) if cum[j] <= d or j == 0)
        i = min(i, len(cum) - 2)
        f = 0.0 if cum[i + 1] == cum[i] else (d - cum[i]) / (cum[i + 1] - cum[i])
        a, b = waypoints[i], waypoints[i + 1]
        return (
            a[0] + f * (b[0] - a[0]),
            a[1] + f * (b[1] - a[1]),
            a[2] + f * (b[2] - a[2]),
            bearing_deg(a[0], a[1], b[0], b[1]),
        )

    pts = []
    prev_alt = None
    prev_t = None
    for k, d in enumerate(dists):
        lat, lon, alt, trk = at(d)
        t = t0 + d / (gs_kt * KT_TO_MS)
        rate = 0.0 if prev_alt is None else (alt - prev_alt) / (t - prev_t) * 60.0
        if prev_alt is None and len(dists) > 1:
            _, _, alt1, _ = at(dists[1])
            rate = (alt1 - alt) / (dists[1] / (gs_kt * KT_TO_MS)) * 60.0
        pts.append(
            Point(
                t=t, lat=lat, lon=lon, alt_ft=alt, gs=gs_kt, track=trk,
                baro_rate=rate, leg_flag=leg_flag and k == 0, **flags,
            )
        )
        prev_alt, prev_t = alt, t
    return pts


def final_approach(end, from_km: float, to_m: float, lateral_m: float = 0.0, **kw) -> list[Point]:
    """Straight-in 3 degree final to a RunwayEnd, from from_km out to to_m short of threshold."""
    back = (end.heading + 180.0) % 360.0
    right = (end.heading + 90.0) % 360.0

    def wp(dist_m: float):
        lat, lon = destination(end.thr_lat, end.thr_lon, back, dist_m)
        if lateral_m:
            lat, lon = destination(lat, lon, right, lateral_m)
        alt = end.elevation_ft + 50 + dist_m * math.tan(math.radians(3)) * 3.28084
        return lat, lon, alt

    return fly([wp(from_km * 1000), wp(to_m)], gs_kt=140.0, **kw)
