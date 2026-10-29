"""Shared builders for synthetic readsb traces."""

from __future__ import annotations


DAY0 = 1793059200.0  # 2026-10-27T00:00:00Z


def pt(
    off: float,
    lat: float = 51.5,
    lon: float = -0.5,
    alt="3000",
    gs: float | None = 200.0,
    track: float | None = 90.0,
    flags: int = 0,
    rate: int | None = 0,
    obj: dict | None = None,
    src: str = "adsb_icao",
    geom: int | None = None,
    geom_rate: int | None = None,
) -> list:
    """One 14-element readsb trace point. ``alt`` is an int, "ground" or None (numeric strings are ints)."""
    if isinstance(alt, str) and alt != "ground":
        alt = int(alt)
    return [off, lat, lon, alt, gs, track, flags, rate, obj, src, geom, geom_rate, None, None]


def trace(points: list[list], icao: str = "a00001", **extra) -> dict:
    return {"icao": icao, "timestamp": DAY0, "trace": points, **extra}

