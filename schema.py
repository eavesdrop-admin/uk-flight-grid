"""Stage A output schema: one Parquet file per UTC day of thinned, pseudonymised UK points.

This module is the contract between ``extract_day.py`` (writer) and the later
stages (segmentation, attribution, rasterisation). Keep it free of heavy imports.

Privacy: no registration, callsign, owner, squawk or type description is ever
written. ``uid`` is a daily-salted hash; the salt is random, held in memory for
one extraction run and then discarded, so the same aircraft has unrelated uids on
different days.
"""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1

# --- Extraction constants (UK-wide; later stages narrow to England and Wales) ---
# Covers Scilly to Shetland and Northern Ireland, plus parts of Ireland and the
# near continent that later stages ignore.
LAT_MIN, LAT_MAX = 49.8, 61.0
LON_MIN, LON_MAX = -8.8, 2.0
ALT_MAX_FT = 10_000          # drop points above this
ALT_MIN_FT = -1_000          # drop corrupt altitudes below this (deep-low baro can read to about -800)
MIN_GS_KT = 40.0             # drop slower points (taxi, hover, parked)
THIN_MIN_DT_S = 8.0          # keep a point if >= 8 s after the last kept one ...
THIN_MIN_DIST_M = 400.0      # ... or >= 400 m away from it
GEOID_OFFSET_FT = 165.0      # AMSL ~= alt_geom (WGS84 height) - 165 ft over the UK

ALT_SRC_GEOM = "geom"
ALT_SRC_BARO = "baro"

# Column order is part of the contract.
COLUMNS: tuple[str, ...] = (
    "uid",
    "leg_flag",
    "t_utc",
    "lat",
    "lon",
    "alt_ft",
    "alt_src",
    "gs",
    "track",
    "baro_rate",
    "category",
    "mil",
    "heli",
)

PARQUET_SCHEMA = pa.schema(
    [
        # sha256(daily_salt || icao)[:8] as a big-endian uint64
        pa.field("uid", pa.uint64(), nullable=False),
        # True when a leg boundary lies between the previous kept point of this
        # aircraft and this one: readsb "new leg" flag, a dropped ground/slow
        # interval, or the aircraft's first kept point of the day.
        pa.field("leg_flag", pa.bool_(), nullable=False),
        pa.field("t_utc", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("lat", pa.float64(), nullable=False),
        pa.field("lon", pa.float64(), nullable=False),
        # Feet above mean sea level (AMSL): alt_geom - GEOID_OFFSET_FT if known, else alt_baro.
        pa.field("alt_ft", pa.int32(), nullable=False),
        pa.field("alt_src", pa.string(), nullable=False),  # "geom" | "baro"
        pa.field("gs", pa.float32(), nullable=False),      # knots
        pa.field("track", pa.float32(), nullable=True),    # degrees true
        pa.field("baro_rate", pa.int32(), nullable=True),  # ft/min (geometric if baro absent)
        # ADS-B emitter category A0..A7 / B* / C*, constant per aircraft per day; null if never broadcast.
        pa.field("category", pa.string(), nullable=True),
        pa.field("mil", pa.bool_(), nullable=False),       # database flag bit 0 (military)
        pa.field("heli", pa.bool_(), nullable=False),      # category == "A7" (rotorcraft)
    ]
)

assert tuple(PARQUET_SCHEMA.names) == COLUMNS

# Columns that must never appear in an output file (checked by tests).
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"icao", "hex", "r", "reg", "registration", "flight", "callsign", "ownOp", "owner", "desc", "t", "type", "year", "squawk"}
)
