# Grid output schema

`build_grid.py` writes four files to `<out>/<window_end>/`:

| File | Rows |
|---|---|
| `cells.parquet` | one per res-9 H3 cell with traffic (sparse) |
| `coverage.parquet` | one per res-7 H3 cell that contains an observed point |
| `airports.parquet` | one per airport with an observed movement or a DfT figure |
| `meta.json` | one build record |

Column types are defined in `grid_schema.py` and checked by the tests. H3 indexes are the
H3 64-bit integer as a signed `Int64` (all valid cells are below 2^63). Altitudes are feet
above mean sea level (Stage A uses `alt_geom - 165 ft`, else barometric). Counts are legs:
one aircraft flight through the area is one leg, however long it stays.

## cells.parquet

"Overhead" is a leg whose path comes within `grid_disk(k=1)` of the cell (about 0.5 km).
"Nearby" is within `grid_disk(k=6)` (about 2 km). Only the part of a leg below 7,000 ft is
counted. A leg counts once per cell, at the lowest altitude it reaches within range.

| Column | Type | Meaning |
|---|---|---|
| `h3_r9` | int64 | cell |
| `oh_day_mean` | float32 | overhead legs per observed day, all bands. Zero days are included in the mean |
| `oh_day_p90` | int16 | 90th percentile of the daily overhead count over all observed days, zero days included |
| `oh_lt1000`, `oh_1000_3000`, `oh_3000_7000` | float32 | mean per day by altitude band (sums to `oh_day_mean`) |
| `oh_night_mean` | float32 | mean per day with local time 23:00 to 06:59 `Europe/London` |
| `oh_alt_p50_ft` | int16, nullable | median altitude of overhead legs, 100 ft bins labelled by centre. Null when there are no overhead legs |
| `oh_days_active` | int16 | observed days with at least one overhead leg |
| `oh_inferred_share` | float32 | share of overhead legs whose only contact with the cell was the synthetic final approach (see `attribute.py`) |
| `nb_day_mean`, `nb_day_p90`, `nb_lt1000`, `nb_1000_3000`, `nb_3000_7000`, `nb_night_mean` | | as above for the 2 km set |
| `top1_airport`, `top1_runway`, `top1_op`, `top1_share` | string, string, string, int16 | most common (airport, runway, op) among overhead legs and its percentage. `op` is `A D C O H M`. Airport and runway are null for overflights, helicopters, military, and where the runway end is unresolved (`runway` only) |
| `top2_*` | | second most common |

A cell is written if `nb_day_mean >= 0.1` or `oh_day_mean >= 0.1`. **A missing cell means
"fewer than 0.1 legs a day", and says nothing about whether the receivers could see.** Read
`coverage.parquet` first.

The day boundary is the UTC day of the Stage A file. The night test uses each leg's local
wall-clock time at its lowest point in range, so it is correct across the March and October
clock changes (tests cover 2026-03-29 and 2026-10-25).

## coverage.parquet

| Column | Type | Meaning |
|---|---|---|
| `h3_r7` | int64 | res-7 cell (about 5 km) |
| `status` | string | `good`, `partial_low_altitude` or `insufficient` |
| `floor_ft` | int16 | 5th percentile altitude of observed points in the cell and its k=1 ring, 100 ft resolution, rounded up |
| `continuity` | float32 | observed aircraft-time / (observed + time lost in gaps over 120 s that span the ring) |
| `sample_points` | int32 | observed points in the ring |

* `good`: `floor_ft <= 1500`, `sample_points >= 500`, and `continuity >= 0.85` (or fewer than 50 leg crossings of the ring).
* `partial_low_altitude`: `sample_points >= 500` and either `1500 < floor_ft <= 4000`, or `floor_ft <= 1500` with `0.60 <= continuity < 0.85`. Counts here are lower bounds.
* `insufficient`: everything else.
* A res-7 cell with no row has no observed point of its own. **Treat it as `insufficient`.** The table is sparse, not dense over England and Wales.
* After classification, cells whose dominant airport has `capture_ratio < 0.85` (see below) move down one step (good to partial, partial to insufficient).

## airports.parquet

| Column | Type | Meaning |
|---|---|---|
| `icao` | string | |
| `name` | string | OurAirports name |
| `observed_movements_annualised` | int32 | observed arrivals + departures by aircraft with emitter category A2 to A5, scaled by 365 / `days_observed` |
| `dft_atm`, `dft_year` | int32, int16, nullable | DfT AVI0102 figure as vendored in `data/dft-aviation/` |
| `capture_ratio` | float32, nullable | `observed_movements_annualised / dft_atm` |
| `top_runways` | string | JSON text for the loader to cast to jsonb: `[{"runway":"27L","op":"A","share":0.31}, ...]`, up to 6, share of the airport's runway-resolved arrivals and departures |

## meta.json

Everything in the plan's `flight_grid_meta` table: `build_id`, `window_start`, `window_end`,
`days_observed`, `days_degraded`, `day_stride`, `builder_version`, `reference_hash`,
`method_url`. `loaded_at` is set by the loader, not the builder. Extra keys: `grid_schema_version`,
`built_at`, `input_days`, `degraded_days`, `data_vintage` (the adsb.lol release tag used for each day),
`source`, `reference` (OurAirports file hashes, airport and runway counts, DfT metadata),
`coverage` (status counts, threshold used, airports under the capture minimum) and `rows`.

## Where this differs from the plan (section 6)

1. **`status` values** are `good`, `partial_low_altitude`, `insufficient`, as in the plan.
2. **`coverage.parquet` is sparse**, not dense over England and Wales. There is no boundary polygon
   in this repo, so rows exist only where the archive observed at least one point in the cell.
   A missing row means `insufficient`. The consumer already has to treat a missing row that way.
3. **"partial" needs enough samples.** The plan's text lets a cell with floor 1,500 to 4,000 ft be
   partial on any sample count. A cell with 20 points is not a lower bound on anything, so it is
   `insufficient` here.
4. **The grid is UK-wide**, not clipped to England and Wales. Stage A extracts the whole UK and the
   reference data holds every GB airport (so a Manchester arrival is not mislabelled an
   overflight); the consumer filters. Airports in the Republic of Ireland and Channel Islands are
   not in the reference, so flights to them are `O`.
5. **Capture ratio uses A2 to A5**, not A3 to A5. A2 holds the ATR and Dash 8 turboprops that
   Southampton, Norwich and Exeter depend on; with A3 to A5 they scored 0.2 to 0.4.
6. **`oh_alt_p50_ft` is nullable** (cells that are only "nearby").
7. **`top_runways`** is JSON text rather than a nested type, and its element shape is this builder's choice.
8. **Time of a pass** for the night split is the time of the leg's lowest sample in range, not the time
   of closest approach to the cell centre.
9. Per-day aggregates are cached in `<cache>/v<BUILDER_VERSION>/<date>/*.parquet` (a directory of
   six files with a `key.json`) rather than one file per day.
10. `validation.json` is not written here; that is the validation harness.
