"""Stage A: stream one adsb.lol day, keep thinned UK airborne points, write one Parquet.

    uv run python extract_day.py --date 2026-10-27 --out out

The day's tar (split into 2 GB parts) is streamed and never stored. The ``heatmap/``
and ``acas/`` entries are skipped with HTTP Range requests, so only the gzip-JSON
``traces/`` entries are downloaded. Each trace is gunzipped and filtered in a worker
process; the survivors are pseudonymised, sorted and written as zstd Parquet with
the columns in ``schema.py``.

Per point we keep: inside the UK bounding box, airborne, not stale, ground speed at
least 40 kt, at most 10,000 ft AMSL, thinned to one point per 8 s or 400 m. Leg
boundaries (readsb "new leg" flag, dropped ground/slow intervals) are carried onto the
next kept point as ``leg_flag``. Registration, callsign, owner, squawk and type are
never read into the output; the ICAO address becomes a daily-salted hash.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import sys
import tarfile
import time
from collections import deque
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

import releases
from remote_tar import SplitTarFile
from schema import (
    ALT_MAX_FT,
    ALT_MIN_FT,
    ALT_SRC_BARO,
    ALT_SRC_GEOM,
    GEOID_OFFSET_FT,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MIN_GS_KT,
    PARQUET_SCHEMA,
    SCHEMA_VERSION,
    THIN_MIN_DIST_M,
    THIN_MIN_DT_S,
)

TRACE_PREFIX = "./traces/"
BATCH_MEMBERS = 256  # traces per worker task (about 10 MB of gzip)
MANIFEST_NAME = "days_manifest.json"
TRAILING_DAYS = 28
DEGRADED_BELOW = 0.60  # degraded when kept points < 60% of the trailing median (i.e. >40% below)
MIN_HISTORY = 7        # trailing-median days needed before we judge a day

# readsb trace point flags (index 6)
FLAG_STALE = 1
FLAG_NEW_LEG = 2
FLAG_ALT_IS_GEOM = 8

M_PER_DEG = 111_320.0


@dataclass
class Stats:
    traces: int = 0            # aircraft in the day (one trace each)
    aircraft_uk: int = 0       # aircraft with at least one point inside the bounding box
    points_total: int = 0
    points_in_bbox: int = 0
    points_kept: int = 0
    points_kept_mlat: int = 0  # kept points whose position source is MLAT

    def add(self, other: Stats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def daily_uid(salt: bytes, icao: str) -> int:
    """First 8 bytes of sha256(salt || icao) as an unsigned 64-bit integer."""
    digest = hashlib.sha256(salt + icao.encode("ascii", "replace")).digest()
    return int.from_bytes(digest[:8], "big")


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def extract_trace(obj: dict, salt: bytes) -> tuple[dict[str, list] | None, Stats]:
    """Filter one decoded ``trace_full_*.json`` into column lists (None if nothing kept)."""
    trace = obj.get("trace") or []
    stats = Stats(traces=1, points_total=len(trace))
    base = _num(obj.get("timestamp"))
    icao = obj.get("icao")
    if base is None or not isinstance(icao, str):
        return None, stats

    db_flags = obj.get("dbFlags")
    mil = bool(db_flags & 1) if isinstance(db_flags, int) else False
    category = None
    uid = daily_uid(salt, icao)

    t_ms: list[int] = []
    lat_l: list[float] = []
    lon_l: list[float] = []
    alt_l: list[int] = []
    src_l: list[str] = []
    gs_l: list[float] = []
    trk_l: list[float | None] = []
    rate_l: list[int | None] = []
    leg_l: list[bool] = []

    pending_break = True  # the aircraft's first kept point starts a leg
    last_t = -1e18
    last_lat = last_lon = 0.0
    seen_uk = False

    for p in trace:
        if len(p) < 12:
            continue
        if category is None:
            extra = p[8]
            if extra:
                category = extra.get("category")
        lat = p[1]
        lon = p[2]
        flags = p[6] or 0
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            if flags & FLAG_NEW_LEG:
                pending_break = True
            continue
        stats.points_in_bbox += 1
        seen_uk = True

        if flags & FLAG_NEW_LEG:
            pending_break = True
        if flags & FLAG_STALE:
            continue

        raw_alt = p[3]
        if raw_alt == "ground":
            pending_break = True
            continue
        gs = _num(p[4])
        if gs is None:
            continue
        if gs < MIN_GS_KT:
            pending_break = True
            continue

        if flags & FLAG_ALT_IS_GEOM:
            geom, baro = _num(raw_alt), None
        else:
            geom, baro = _num(p[10]), _num(raw_alt)
        if geom is not None:
            alt = geom - GEOID_OFFSET_FT
            alt_src = ALT_SRC_GEOM
        elif baro is not None:
            alt = baro
            alt_src = ALT_SRC_BARO
        else:
            continue
        if alt > ALT_MAX_FT or alt < ALT_MIN_FT:
            continue

        t = base + p[0]
        if not pending_break:
            dt = t - last_t
            if dt < THIN_MIN_DT_S:
                dy = (lat - last_lat) * M_PER_DEG
                dx = (lon - last_lon) * M_PER_DEG * math.cos(math.radians(lat))
                if dx * dx + dy * dy < THIN_MIN_DIST_M * THIN_MIN_DIST_M:
                    continue

        rate = _num(p[7])
        if rate is None:
            rate = _num(p[11])
        track = _num(p[5])

        t_ms.append(round(t * 1000))
        lat_l.append(lat)
        lon_l.append(lon)
        alt_l.append(round(alt))
        src_l.append(alt_src)
        gs_l.append(gs)
        trk_l.append(track)
        rate_l.append(None if rate is None else round(rate))
        leg_l.append(pending_break)
        if p[9] == "mlat":
            stats.points_kept_mlat += 1
        pending_break = False
        last_t, last_lat, last_lon = t, lat, lon

    if seen_uk:
        stats.aircraft_uk = 1
    n = len(t_ms)
    stats.points_kept = n
    if n == 0:
        return None, stats

    cols = {
        "uid": [uid] * n,
        "leg_flag": leg_l,
        "t_utc": t_ms,
        "lat": lat_l,
        "lon": lon_l,
        "alt_ft": alt_l,
        "alt_src": src_l,
        "gs": gs_l,
        "track": trk_l,
        "baro_rate": rate_l,
        "category": [category] * n,
        "mil": [mil] * n,
        "heli": [category == "A7"] * n,
    }
    return cols, stats


def _decode(raw: bytes) -> dict:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return orjson.loads(raw)


def process_batch(members: list[bytes], salt: bytes) -> tuple[pa.Table | None, Stats]:
    """Decode and filter a batch of raw trace files into one Arrow table."""
    total = Stats()
    parts: list[dict[str, list]] = []
    for raw in members:
        try:
            obj = _decode(raw)
        except (OSError, ValueError):
            continue  # corrupt member: skip, it shows up as a lower aircraft count
        cols, stats = extract_trace(obj, salt)
        total.add(stats)
        if cols is not None:
            parts.append(cols)
    if not parts:
        return None, total
    merged = {name: [] for name in parts[0]}
    for cols in parts:
        for name, values in cols.items():
            merged[name].extend(values)
    return table_from_columns(merged), total


def table_from_columns(cols: dict[str, list]) -> pa.Table:
    arrays = [pa.array(cols[f.name], type=f.type) for f in PARQUET_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)


_WORKER_SALT = b""


def _init_worker(salt: bytes) -> None:
    global _WORKER_SALT
    _WORKER_SALT = salt


def _work(members: list[bytes]) -> tuple[pa.Table | None, Stats]:
    return process_batch(members, _WORKER_SALT)


def iter_trace_batches(tar: tarfile.TarFile, batch_members: int = BATCH_MEMBERS):
    """Yield lists of raw trace-file bytes, skipping every non-trace entry unread."""
    batch: list[bytes] = []
    for member in tar:
        if not member.isfile() or not member.name.startswith(TRACE_PREFIX):
            continue
        fh = tar.extractfile(member)
        if fh is None:
            continue
        batch.append(fh.read())
        if len(batch) >= batch_members:
            yield batch
            batch = []
    if batch:
        yield batch


def run_extraction(
    fileobj,
    salt: bytes,
    workers: int,
    batch_members: int = BATCH_MEMBERS,
) -> tuple[pa.Table, Stats]:
    """Stream ``fileobj`` (a tar) through ``workers`` processes; return the sorted table."""
    tables: list[pa.Table] = []
    total = Stats()

    def collect(result: tuple[pa.Table | None, Stats]) -> None:
        table, stats = result
        total.add(stats)
        if table is not None:
            tables.append(table)

    with tarfile.open(fileobj=fileobj, mode="r:") as tar:
        if workers <= 1:
            for batch in iter_trace_batches(tar, batch_members):
                collect(process_batch(batch, salt))
        else:
            ctx = multiprocessing.get_context("spawn")
            pending: deque = deque()
            with ctx.Pool(workers, initializer=_init_worker, initargs=(salt,)) as pool:
                for batch in iter_trace_batches(tar, batch_members):
                    pending.append(pool.apply_async(_work, (batch,)))
                    while len(pending) >= workers * 3:  # bounds raw bytes held in memory
                        collect(pending.popleft().get())
                while pending:
                    collect(pending.popleft().get())

    if tables:
        table = pa.concat_tables(tables).sort_by([("uid", "ascending"), ("t_utc", "ascending")])
    else:
        table = PARQUET_SCHEMA.empty_table()
    return table, total


def write_parquet(table: pa.Table, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table.replace_schema_metadata({k: str(v) for k, v in metadata.items()}),
        tmp,
        compression="zstd",
        compression_level=9,
        row_group_size=500_000,
        use_dictionary=["alt_src", "category"],
    )
    os.replace(tmp, path)


def evaluate_manifest(days: dict[str, dict]) -> None:
    """Mark each day ``ok`` or ``degraded`` against the trailing 28-day median of kept points.

    Degraded days are excluded from the median so one bad day cannot hide the next.
    """
    good: dict[date, int] = {}
    for key in sorted(days):
        entry = days[key]
        day = date.fromisoformat(key)
        history = [v for d, v in good.items() if 0 < (day - d).days <= TRAILING_DAYS]
        status = "ok"
        if len(history) >= MIN_HISTORY and entry["points_kept"] < DEGRADED_BELOW * statistics.median(history):
            status = "degraded"
        entry["status"] = status
        if status == "ok":
            good[day] = entry["points_kept"]


def update_manifest(path: Path, key: str, entry: dict) -> dict:
    manifest = {"schema_version": SCHEMA_VERSION, "days": {}}
    if path.exists():
        manifest = json.loads(path.read_text())
    manifest["days"][key] = entry
    evaluate_manifest(manifest["days"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return manifest["days"][key]


def extract_day(
    day: date,
    out_dir: Path,
    *,
    parts: list[tuple[str, int]] | None = None,
    release: releases.Release | None = None,
    workers: int | None = None,
    salt: bytes | None = None,
) -> dict:
    """Extract ``day`` to ``out_dir/extract/<year>/<date>.parquet``; returns the manifest entry.

    Give ``parts`` (local tar pieces or URLs with exact sizes) to bypass release discovery.
    """
    if parts is None:
        release = release or releases.resolve_day(day)
        if release is None:
            raise SystemExit(f"no usable adsb.lol release found for {day}")
        parts = release.parts()
    workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 2))
    # Random per run and never written anywhere: uids cannot be linked across days.
    salt = salt if salt is not None else os.urandom(32)

    started = time.time()
    source = SplitTarFile(parts)
    try:
        table, stats = run_extraction(source, salt, workers)
    finally:
        fetched = source.bytes_fetched
        source.close()

    out_path = out_dir / "extract" / f"{day:%Y}" / f"{day.isoformat()}.parquet"
    write_parquet(
        table,
        out_path,
        {
            "schema_version": SCHEMA_VERSION,
            "date": day.isoformat(),
            "release_tag": release.tag if release else "",
            "source": "adsb.lol globe_history (ODbL 1.0)",
        },
    )
    entry = {
        "date": day.isoformat(),
        "release": release.tag if release else None,
        "repo": release.repo if release else None,
        "release_kind": release.kind if release else None,
        "release_bytes": sum(size for _, size in parts),
        "bytes_fetched": fetched,
        "aircraft": stats.traces,
        "aircraft_uk": stats.aircraft_uk,
        "distinct_uids": int(len(table.column("uid").unique())),
        "points_total": stats.points_total,
        "points_in_bbox": stats.points_in_bbox,
        "points_kept": stats.points_kept,
        "points_kept_mlat": stats.points_kept_mlat,
        "output_bytes": out_path.stat().st_size,
        "seconds": round(time.time() - started, 1),
        "schema_version": SCHEMA_VERSION,
        "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return update_manifest(out_dir / MANIFEST_NAME, day.isoformat(), entry)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, type=date.fromisoformat, help="UTC day, YYYY-MM-DD")
    ap.add_argument("--out", default="out", type=Path, help="output directory (default: out)")
    ap.add_argument("--workers", type=int, default=None, help="decode processes (default: cores - 2, max 8)")
    ap.add_argument(
        "--tar",
        nargs="+",
        default=None,
        metavar="PART",
        help="local tar pieces in order (skips release discovery; for re-runs and tests)",
    )
    args = ap.parse_args(argv)

    parts = None
    if args.tar:
        parts = [(p, os.path.getsize(p)) for p in args.tar]
    entry = extract_day(args.date, args.out, parts=parts, workers=args.workers)
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
