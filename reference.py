"""Airport and runway-end reference data.

Sources:
  * OurAirports airports.csv / runways.csv (public domain).
  * DfT Aviation Statistics AVI0102b air transport movements (OGL v3.0),
    vendored at data/dft-aviation/avi0102.json.

Runway coordinates in OurAirports are the pavement ends. When a runway end has
a displaced threshold the landing threshold sits that far inboard along the
runway heading; departures start the roll at the pavement end.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from geo import M_PER_FT, bearing_deg, destination, haversine_m

OURAIRPORTS_URLS = {
    "airports.csv": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "runways.csv": "https://davidmegginson.github.io/ourairports-data/runways.csv",
}
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DFT_PATH = DATA_DIR / "dft-aviation" / "avi0102.json"

# England and Wales only for v1 (plan section 5). Scottish and Northern Irish
# airports within 30 km of the border are not included by this filter.
DEFAULT_REGIONS = ("GB-ENG", "GB-WLS")
AIRPORT_TYPES = ("large_airport", "medium_airport", "small_airport")

_IDENT_RE = re.compile(r"^(\d{1,2})([LRC]?)$")


@dataclass(frozen=True)
class RunwayEnd:
    airport: str  # ICAO
    ident: str  # e.g. "27L"
    lat: float  # pavement end (start of take-off roll)
    lon: float
    thr_lat: float  # landing threshold (displaced-threshold adjusted)
    thr_lon: float
    elevation_ft: float
    heading: float  # true heading of travel when using this end, degrees
    displaced_m: float
    length_m: float
    opposite: str  # ident of the reciprocal end

    @property
    def key(self) -> tuple[str, str]:
        return (self.airport, self.ident)


@dataclass
class Airport:
    icao: str
    name: str
    iata: str | None
    lat: float
    lon: float
    elevation_ft: float
    type: str
    region: str
    dft_atm: int | None = None  # annual air transport movements (DfT AVI0102b)
    dft_year: int | None = None
    runways: list[RunwayEnd] = field(default_factory=list)


@dataclass
class Reference:
    airports: dict[str, Airport]
    dft_source: dict = field(default_factory=dict)

    @property
    def runway_ends(self) -> list[RunwayEnd]:
        return [r for a in self.airports.values() for r in a.runways]

    def runway_end(self, airport: str, ident: str) -> RunwayEnd | None:
        a = self.airports.get(airport)
        if a is None:
            return None
        return next((r for r in a.runways if r.ident == ident), None)

    def ends_near(self, lat: float, lon: float, radius_m: float) -> list[RunwayEnd]:
        """Runway ends whose landing threshold is within radius_m of the point."""
        # Cheap degree box first, then exact distance.
        dlat = radius_m / 111_000.0 + 0.01
        out = []
        for r in self.runway_ends:
            if abs(r.thr_lat - lat) > dlat:
                continue
            if haversine_m(lat, lon, r.thr_lat, r.thr_lon) <= radius_m:
                out.append(r)
        return out

    def starts_near(self, lat: float, lon: float, radius_m: float) -> list[RunwayEnd]:
        """Runway ends whose take-off start (pavement end) is within radius_m."""
        dlat = radius_m / 111_000.0 + 0.01
        out = []
        for r in self.runway_ends:
            if abs(r.lat - lat) > dlat:
                continue
            if haversine_m(lat, lon, r.lat, r.lon) <= radius_m:
                out.append(r)
        return out

    def content_hash(self) -> str:
        """Deterministic hash of everything attribution depends on."""
        payload = [
            {
                "icao": a.icao,
                "atm": a.dft_atm,
                "year": a.dft_year,
                "ends": [
                    [r.ident, round(r.lat, 6), round(r.lon, 6), round(r.thr_lat, 6),
                     round(r.thr_lon, 6), round(r.elevation_ft), round(r.heading, 2)]
                    for r in sorted(a.runways, key=lambda r: r.ident)
                ],
            }
            for a in sorted(self.airports.values(), key=lambda a: a.icao)
        ]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def download_ourairports(dest: Path) -> dict[str, Path]:
    """Fetch the two OurAirports CSVs. Network; never called at test time."""
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in OURAIRPORTS_URLS.items():
        path = dest / name
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
            path.write_bytes(resp.read())
        out[name] = path
    return out


def _f(v: str | None) -> float | None:
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _opposite_ident(ident: str) -> str:
    m = _IDENT_RE.match(ident)
    if not m:
        return ""
    num = (int(m.group(1)) + 18 - 1) % 36 + 1
    side = {"L": "R", "R": "L", "C": "C", "": ""}[m.group(2)]
    return f"{num:02d}{side}"


def load_dft(path: Path = DEFAULT_DFT_PATH) -> tuple[dict[str, dict], dict]:
    """DfT AVI0102b rows keyed by ICAO, plus the sibling metadata (if any)."""
    rows = json.loads(Path(path).read_text())
    meta_path = Path(path).with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {r["icao"]: r for r in rows if r.get("icao")}, meta


def _runway_ends(row: dict, airport: Airport) -> list[RunwayEnd]:
    if row.get("closed") == "1":
        return []
    if (row.get("surface") or "").upper().startswith("WATER"):
        return []
    ends = {}
    for side in ("le", "he"):
        ident = (row.get(f"{side}_ident") or "").strip()
        lat, lon = _f(row.get(f"{side}_latitude_deg")), _f(row.get(f"{side}_longitude_deg"))
        if not ident or ident.upper().startswith("H") or lat is None or lon is None:
            continue
        ends[side] = (ident, lat, lon)
    if not ends:
        return []

    other = {"le": "he", "he": "le"}
    length_ft = _f(row.get("length_ft"))
    out = []
    for side, (ident, lat, lon) in ends.items():
        o = ends.get(other[side])
        if o is not None:
            heading = bearing_deg(lat, lon, o[1], o[2])
            length_m = haversine_m(lat, lon, o[1], o[2])
        else:
            heading = _f(row.get(f"{side}_heading_degT"))
            if heading is None:
                continue
            length_m = (length_ft or 0.0) * M_PER_FT
        displaced_m = (_f(row.get(f"{side}_displaced_threshold_ft")) or 0.0) * M_PER_FT
        thr_lat, thr_lon = destination(lat, lon, heading, displaced_m) if displaced_m else (lat, lon)
        elev = _f(row.get(f"{side}_elevation_ft"))
        out.append(
            RunwayEnd(
                airport=airport.icao,
                ident=ident,
                lat=lat,
                lon=lon,
                thr_lat=thr_lat,
                thr_lon=thr_lon,
                elevation_ft=airport.elevation_ft if elev is None else elev,
                heading=heading,
                displaced_m=displaced_m,
                length_m=length_m,
                opposite=_opposite_ident(ident),
            )
        )
    return out


def build_reference(
    airports_csv: Path,
    runways_csv: Path,
    dft_json: Path | None = DEFAULT_DFT_PATH,
    regions: tuple[str, ...] | None = DEFAULT_REGIONS,
    include_small_without_iata: bool = False,
) -> Reference:
    """Join OurAirports airports and runways to DfT movements.

    Keeps GB airports of type large/medium, plus small airports that have an
    IATA code (unless include_small_without_iata). Airports without any usable
    runway end are dropped. `regions=None` keeps every GB region.
    """
    dft, dft_meta = load_dft(dft_json) if dft_json else ({}, {})

    airports: dict[str, Airport] = {}
    by_ref: dict[str, Airport] = {}
    with open(airports_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["iso_country"] != "GB" or row["type"] not in AIRPORT_TYPES:
                continue
            if regions is not None and row["iso_region"] not in regions:
                continue
            iata = row["iata_code"].strip() or None
            if row["type"] == "small_airport" and not iata and not include_small_without_iata:
                continue
            lat, lon = _f(row["latitude_deg"]), _f(row["longitude_deg"])
            if lat is None or lon is None:
                continue
            # OurAirports' ident is a local code ("GB-1222") or an outdated ICAO for some
            # fields (RAF Wyton, St Athan); the ICAO code, then the GPS code, is the real one.
            icao = (row.get("icao_code") or "").strip() or (row.get("gps_code") or "").strip() or row["ident"]
            a = Airport(
                icao=icao,
                name=row["name"],
                iata=iata,
                lat=lat,
                lon=lon,
                elevation_ft=_f(row["elevation_ft"]) or 0.0,
                type=row["type"],
                region=row["iso_region"],
            )
            d = dft.get(icao)
            if d is not None:
                a.dft_atm = int(d["annual_movements"])
                a.dft_year = int(d["data_year"])
            airports[icao] = a
            by_ref[row["id"]] = a

    with open(runways_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            a = by_ref.get(row["airport_ref"])
            if a is not None:
                a.runways.extend(_runway_ends(row, a))

    airports = {k: a for k, a in airports.items() if a.runways}
    return Reference(airports=airports, dft_source=dft_meta)


__all__ = [
    "Airport",
    "Reference",
    "RunwayEnd",
    "build_reference",
    "download_ourairports",
    "load_dft",
]
