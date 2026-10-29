"""Print what a built grid says about a handful of named places.

    uv run python sanity.py out/grid/2026-10-27

Overhead means flights within about 0.5 km, nearby within about 2 km. A place with no cell
row has fewer than 0.1 overhead and nearby legs a day; whether that means "quiet" or
"unobserved" is what the coverage column says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from h3.api import basic_int as h3i

PLACES = [
    ("Hounslow TW3 (EGLL 27L final)", 51.46830, -0.36396),
    ("Isleworth TW7 (EGLL 27L final)", 51.47286, -0.33604),
    ("Cranford TW5 (EGLL)", 51.48247, -0.38629),
    ("Windsor SL4 (EGLL 09 final)", 51.47534, -0.62761),
    ("Horley RH6 (Gatwick)", 51.17351, -0.16464),
    ("Heald Green M22 (Manchester)", 53.38577, -2.26013),
    ("Keynsham BS31 2FE (Bristol)", 51.42118, -2.49510),
    ("Southampton SO17 3TF", 50.93756, -1.38780),
    ("Rhayader LD6 (mid-Wales)", 52.31050, -3.50896),
    ("Leeds LS1 (city centre)", 53.79763, -1.55126),
]


def main(grid_dir: Path) -> None:
    cells = pl.read_parquet(grid_dir / "cells.parquet")
    cover = pl.read_parquet(grid_dir / "coverage.parquet")
    meta = json.loads((grid_dir / "meta.json").read_text())
    print(f"build {meta['build_id']}: {meta['days_observed']} days, stride {meta['day_stride']}\n")
    hdr = f"{'place':34} {'<3k':>6} {'<7k':>6} {'day':>6} {'night':>6} {'nb<7k':>6}  {'dominant':18} {'cov':>22} {'floor':>6} {'cont':>5}"
    print(hdr)
    for name, lat, lon in PLACES:
        c9 = h3i.latlng_to_cell(lat, lon, 9)
        c7 = h3i.cell_to_parent(c9, 7)
        row = cells.filter(pl.col("h3_r9") == c9)
        cv = cover.filter(pl.col("h3_r7") == c7)
        status = cv["status"][0] if cv.height else "no row (insufficient)"
        floor = f"{cv['floor_ft'][0]}" if cv.height else "-"
        cont = f"{cv['continuity'][0]:.2f}" if cv.height else "-"
        if row.height:
            r = row.row(0, named=True)
            low = r["oh_lt1000"] + r["oh_1000_3000"]
            dom = f"{r['top1_op']} {r['top1_airport'] or ''} {r['top1_runway'] or ''} {r['top1_share']}%".replace("  ", " ")
            print(f"{name:34} {low:6.1f} {r['oh_day_mean']:6.1f} {r['oh_day_mean'] - r['oh_night_mean']:6.1f} {r['oh_night_mean']:6.1f} {r['nb_day_mean']:6.1f}  {dom:18} {status:>22} {floor:>6} {cont:>5}")
        else:
            print(f"{name:34} {'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6}  {'(no cell row)':18} {status:>22} {floor:>6} {cont:>5}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
