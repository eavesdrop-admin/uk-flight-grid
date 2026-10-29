"""Cut the small trace fixture used by the tests from a real adsb.lol day.

    uv run python tests/make_fixtures.py --date 2026-10-27

Streams the day (heatmap skipped), picks one or two traces for each scenario the
extractor has to handle, trims them to the part that matters, strips everything
identifying (registration, callsign, owner, type, squawk, real ICAO address) and writes
``tests/fixtures/traces_<date>.json``. The ICAO addresses become ``f00000``, ``f00001``...

The traces are derived from adsb.lol data, made available under the ODbL 1.0.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import date
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import extract_day as ex  # noqa: E402
import releases  # noqa: E402
from remote_tar import SplitTarFile  # noqa: E402
from schema import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN  # noqa: E402

EGLL = (51.4775, -0.4614)
CONTEXT = 25     # points kept either side of the interesting window
MAX_POINTS = 400
WANT = {
    "helicopter": 2,
    "military": 2,
    "heathrow_arrival": 2,
    "heathrow_departure": 2,
    "ground_ops": 2,
    "stale_points": 1,
    "baro_only": 2,
    "mlat_points": 1,
    "cruise_only": 2,
    "outside_uk": 2,
    "multi_leg": 2,
}


def in_box(p) -> bool:
    return LAT_MIN <= p[1] <= LAT_MAX and LON_MIN <= p[2] <= LON_MAX


def near_egll(p, deg=0.12) -> bool:
    return abs(p[1] - EGLL[0]) < deg and abs(p[2] - EGLL[1]) < deg * 1.6


def classify(obj: dict) -> list[str]:
    tr = obj["trace"]
    box = [p for p in tr if in_box(p)]
    tags: list[str] = []
    if not box:
        return ["outside_uk"] if len(tr) > 60 else []
    air = [p for p in box if p[3] != "ground" and isinstance(p[4], (int, float)) and p[4] >= 40]
    low_air = [p for p in air if isinstance(p[3], int) and p[3] < 9000]
    cat = next((p[8].get("category") for p in tr if p[8] and p[8].get("category")), None)
    if cat == "A7" and len(low_air) >= 20:
        tags.append("helicopter")
    if (obj.get("dbFlags") or 0) & 1 and len(low_air) >= 20 and cat != "A7":
        tags.append("military")
    low_egll = [p for p in box if near_egll(p) and isinstance(p[3], int) and p[3] < 3000]
    if len(low_egll) >= 10 and any(isinstance(p[7], int) and p[7] < -300 for p in low_egll):
        tags.append("heathrow_arrival")
    if len(low_egll) >= 10 and any(isinstance(p[7], int) and p[7] > 500 for p in low_egll):
        tags.append("heathrow_departure")
    if any(p[3] == "ground" for p in box) and air:
        tags.append("ground_ops")
    if any(p[6] & 1 for p in air):
        tags.append("stale_points")
    if len(air) >= 20 and all(p[10] is None for p in air) and all(isinstance(p[3], int) for p in air):
        tags.append("baro_only")
    if any(p[9] == "mlat" for p in air):
        tags.append("mlat_points")
    if air and all(isinstance(p[3], int) and p[3] > 12_000 for p in air):
        tags.append("cruise_only")
    if sum(1 for p in box if p[6] & 2) >= 2 and len(air) >= 20:
        tags.append("multi_leg")
    return tags


def window(tr: list, tag: str) -> list:
    """The slice of the trace worth keeping for ``tag``."""
    idx = [i for i, p in enumerate(tr) if in_box(p)]
    if tag == "outside_uk":
        return tr[:60]
    if not idx:
        return []
    if tag.startswith("heathrow"):
        near = [i for i in idx if near_egll(tr[i])]
        idx = near or idx
    lo, hi = max(0, idx[0] - CONTEXT), min(len(tr), idx[-1] + CONTEXT + 1)
    if hi - lo > MAX_POINTS:
        mid = (idx[0] + idx[-1]) // 2
        lo, hi = max(0, mid - MAX_POINTS // 2), min(len(tr), mid + MAX_POINTS // 2)
    return tr[lo:hi]


def sanitise(obj: dict, tag: str, n: int) -> dict:
    cat = next((p[8].get("category") for p in obj["trace"] if p[8] and p[8].get("category")), None)
    pts = []
    for p in window(obj["trace"], tag):
        q = list(p)
        q[8] = {"category": cat} if q[8] and cat else None
        pts.append(q)
    out = {
        "icao": f"f{n:05x}",
        "timestamp": obj["timestamp"],
        "trace": pts,
        "_scenario": tag,
    }
    if obj.get("dbFlags") is not None:
        out["dbFlags"] = obj["dbFlags"]
    return out


def produces_expected_output(candidate: dict, tag: str) -> bool:
    """The trimmed trace must actually exercise the extractor the way ``tag`` promises."""
    cols, _ = ex.extract_trace(candidate, b"\0" * 32)
    if tag in ("outside_uk", "cruise_only"):
        return cols is None
    if cols is None:
        return False
    if tag in ("multi_leg", "ground_ops"):
        return sum(cols["leg_flag"]) >= 2
    if tag == "baro_only":
        return set(cols["alt_src"]) == {"baro"}
    if tag == "helicopter":
        return all(cols["heli"])
    if tag == "military":
        return all(cols["mil"])
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, type=date.fromisoformat)
    args = ap.parse_args()

    release = releases.resolve_day(args.date)
    if release is None:
        raise SystemExit("day not found")
    src = SplitTarFile(release.parts())
    need = dict(WANT)
    picked: list[dict] = []
    with tarfile.open(fileobj=src, mode="r:") as tar:
        for member in tar:
            if not member.isfile() or not member.name.startswith(ex.TRACE_PREFIX):
                continue
            obj = ex._decode(tar.extractfile(member).read())
            for tag in classify(obj):
                if need.get(tag, 0) > 0:
                    candidate = sanitise(obj, tag, len(picked))
                    if not produces_expected_output(candidate, tag):
                        continue
                    need[tag] -= 1
                    picked.append(candidate)
                    break
            if not any(need.values()):
                break
    print("unfilled:", {k: v for k, v in need.items() if v}, "fetched MB:", round(src.bytes_fetched / 1e6))
    out = ROOT / "tests" / "fixtures" / f"traces_{args.date.isoformat()}.json"
    payload = {
        "_source": "adsb.lol globe_history, ODbL 1.0; identifiers stripped, ICAO addresses replaced",
        "_date": args.date.isoformat(),
        "traces": picked,
    }
    out.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, {len(picked)} traces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
