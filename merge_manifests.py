"""Merge the per-day manifest files Stage A jobs wrote into one ``days_manifest.json``.

    python merge_manifests.py out/extract/manifests out/days_manifest.json

Each CI extraction job sees only its own day, so it cannot judge that day against the
trailing 28-day median. This re-runs that judgement (extract_day.evaluate_manifest) over
every day so build_grid.py excludes degraded days.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from extract_day import evaluate_manifest
from schema import SCHEMA_VERSION


def merge(src_dir: Path) -> dict:
    days: dict[str, dict] = {}
    for p in sorted(Path(src_dir).glob("*.json")):
        days.update(json.loads(p.read_text()).get("days", {}))
    evaluate_manifest(days)
    return {"schema_version": SCHEMA_VERSION, "days": days}


def main(argv: list[str] | None = None) -> int:
    src, dst = (Path(x) for x in (sys.argv[1:] if argv is None else argv))
    manifest = merge(src)
    dst.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    degraded = [d for d, e in manifest["days"].items() if e["status"] == "degraded"]
    print(f"{len(manifest['days'])} days, {len(degraded)} degraded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
