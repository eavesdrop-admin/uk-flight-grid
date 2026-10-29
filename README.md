# uk-flight-grid

Builds an aggregated UK overflight exposure grid from the [adsb.lol](https://adsb.lol)
historical ADS-B archive: for each H3 cell, roughly how many aircraft pass overhead per
day below 7,000 ft, split by altitude band, day and night, and which airport and runway
they belong to.

This repository is the published build method. It is released so that anyone using the
resulting grid can see exactly how it was made, as the Open Database License (ODbL
section 4.6(b)) asks of a derivative database. The code is MIT licensed. The grid itself is
not published here.

## Data and licence

* Source data: the `adsblol/globe_history_<year>` GitHub release archives, made available
  by adsb.lol under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  Contains information from adsb.lol.
* Airport and runway reference data: [OurAirports](https://ourairports.com/data/) (public domain).
* Code in this repository: MIT, see `LICENSE`.

No registration, callsign, owner, squawk or aircraft type is read into any output. The
ICAO address is replaced by `sha256(daily_salt || icao)[:8 bytes]`; the salt is random per
run, held in memory and never written, so the same aircraft cannot be linked across days.

## Pipeline

1. **Stage A, `extract_day.py`** (this stage): streams one day's tar from GitHub, skips the
   `heatmap/` and `acas/` entries with HTTP Range requests (about a quarter of the bytes),
   decodes `traces/*.json.gz` in worker processes, and writes one Parquet file per UTC day.
2. **Stage B, `build_grid.py`**: segments tracks into legs (`segment.py`), attributes them to
   airports and runways (`attribute.py`, `reference.py`), rasterises to H3 res 9 (`rasterise.py`),
   aggregates per day and merges across days (`aggregate.py`), and judges receiver coverage
   (`coverage.py`). Output tables are described in [SCHEMA.md](SCHEMA.md).

### What Stage A keeps

Per point: inside the UK bounding box (lat 49.8 to 61.0, lon -8.8 to 2.0), airborne, not
flagged stale, ground speed at least 40 kt, altitude between -1,000 and 10,000 ft above
mean sea level, thinned to one point per 8 s or 400 m. Altitude is `alt_geom - 165 ft`
when available (WGS84 height to approximate AMSL), otherwise barometric altitude; the
choice is recorded in `alt_src`. MLAT-only aircraft (the `mlatonly` release) are not used.
Leg boundaries (readsb "new leg" flag, or a dropped ground/slow interval) are carried onto
the next kept point as `leg_flag`.

Output columns are defined in `schema.py` (`PARQUET_SCHEMA`): `uid, leg_flag, t_utc, lat,
lon, alt_ft, alt_src, gs, track, baro_rate, category, mil, heli`.

### Which release is used

`releases.py` looks for `v<YYYY>.<MM>.<DD>-planes-readsb-<kind>-0` in the repo for the
day's year and then the neighbouring years (the yearly repos overlap at New Year). `prod`
is preferred; `staging` is used if `prod` is missing or more than 15% smaller. Set
`GITHUB_TOKEN` to avoid the unauthenticated API rate limit.

## Usage

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python extract_day.py --date 2026-10-27 --out out     # about 6 minutes, 3 GB over the wire
uv run python extract_day.py --date 2026-10-27 --tar day.tar.aa day.tar.ab   # from local parts
uv run pytest
uv run python build_grid.py --extract-dir out/extract --out out/grid   # about a minute for three days
uv run python sanity.py out/grid/<window_end>                         # what the grid says about named places
```

`build_grid.py` caches each day under `out/agg/v<BUILDER_VERSION>/`; the cache is reused while the
extract file, the reference data and `BUILDER_VERSION` are unchanged. Bump `BUILDER_VERSION` in
`build_grid.py` whenever a change can alter any output number. OurAirports files are downloaded once
into `data/raw/ourairports/`.

Output: `out/extract/<year>/<date>.parquet` and `out/days_manifest.json` (release used,
bytes, aircraft and point counts, and an `ok`/`degraded` status against the trailing
28-day median). `out/` is gitignored.

`tests/make_fixtures.py` regenerates `tests/fixtures/traces_<date>.json`, about 20 anonymised
real traces covering the awkward cases (ground points, stale points, helicopters, military,
barometric-only altitude, MLAT points, aircraft that never enter the UK).
