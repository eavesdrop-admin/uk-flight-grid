"""Stage B output schema: the grid tables and their column types.

This is the contract with whatever loads the grid into a database. Column names and
meanings follow the build plan (section 6); see SCHEMA.md for definitions and for the
few places this builder is stricter than the plan. Keep this module free of heavy imports.
"""

from __future__ import annotations

import polars as pl

GRID_SCHEMA_VERSION = 1

# --- Altitude bands (feet above mean sea level) and night ---
BAND_EDGES_FT = (1_000, 3_000, 7_000)  # <1,000 | 1,000-3,000 | 3,000-7,000; 7,000 and above is not counted
MAX_COUNTED_ALT_FT = 7_000
NIGHT_START_HOUR = 23  # 23:00 to 07:00 Europe/London
NIGHT_END_HOUR = 7
LOCAL_TZ = "Europe/London"

# --- Rasterisation ---
CELL_RES = 9
COVERAGE_RES = 7
OVERHEAD_K = 1  # grid_disk(k=1) around a path cell: about 0.5 km
NEARBY_K = 6  # grid_disk(k=6): about 2 km
MIN_MEAN_PER_DAY = 0.1  # a cell is kept if nb_day_mean >= this or oh_day_mean >= this

_TOP = {
    "{p}_airport": pl.String,  # ICAO, null for overflights / helicopters / military
    "{p}_runway": pl.String,  # runway end, e.g. "27L"; null when unresolved or not an airport movement
    "{p}_op": pl.String,  # one of A D C O H M
    "{p}_share": pl.Int16,  # percent of the cell's overhead legs
}

CELLS_SCHEMA: dict[str, pl.DataType] = {
    "h3_r9": pl.Int64,
    "oh_day_mean": pl.Float32,
    "oh_day_p90": pl.Int16,
    "oh_lt1000": pl.Float32,
    "oh_1000_3000": pl.Float32,
    "oh_3000_7000": pl.Float32,
    "oh_night_mean": pl.Float32,
    "oh_alt_p50_ft": pl.Int16,
    "oh_days_active": pl.Int16,
    "oh_inferred_share": pl.Float32,
    "nb_day_mean": pl.Float32,
    "nb_day_p90": pl.Int16,
    "nb_lt1000": pl.Float32,
    "nb_1000_3000": pl.Float32,
    "nb_3000_7000": pl.Float32,
    "nb_night_mean": pl.Float32,
    **{k.format(p="top1"): v for k, v in _TOP.items()},
    **{k.format(p="top2"): v for k, v in _TOP.items()},
}

STATUS_GOOD = "good"
STATUS_PARTIAL = "partial_low_altitude"
STATUS_INSUFFICIENT = "insufficient"
COVERAGE_STATUSES = (STATUS_GOOD, STATUS_PARTIAL, STATUS_INSUFFICIENT)

COVERAGE_SCHEMA: dict[str, pl.DataType] = {
    "h3_r7": pl.Int64,
    "status": pl.String,
    "floor_ft": pl.Int16,
    "continuity": pl.Float32,
    "sample_points": pl.Int32,
}

AIRPORTS_SCHEMA: dict[str, pl.DataType] = {
    "icao": pl.String,
    "name": pl.String,
    "observed_movements_annualised": pl.Int32,
    "dft_atm": pl.Int32,
    "dft_year": pl.Int16,
    "capture_ratio": pl.Float32,
    "top_runways": pl.String,  # JSON text; the loader casts it to jsonb
}

META_REQUIRED_KEYS = (
    "build_id",
    "window_start",
    "window_end",
    "days_observed",
    "days_degraded",
    "day_stride",
    "builder_version",
    "reference_hash",
    "method_url",
)


def conform(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select and cast to exactly `schema`, in order."""
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])
