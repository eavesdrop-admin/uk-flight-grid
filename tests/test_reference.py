from pathlib import Path

import pytest

from reference import DEFAULT_DFT_PATH, build_reference

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ref():
    return build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", DEFAULT_DFT_PATH)


def test_egll_ident_mapping_from_real_runways_csv(ref):
    egll = ref.airports["EGLL"]
    assert {r.ident for r in egll.runways} == {"09L", "09R", "27L", "27R"}
    e = {r.ident: r for r in egll.runways}
    # 27L is the southern runway's eastern end; 27R the northern runway's.
    assert e["27L"].lat < e["27R"].lat
    assert e["09R"].lat < e["09L"].lat
    assert e["27L"].lat == pytest.approx(51.464957, abs=1e-5)
    assert e["27R"].lat == pytest.approx(51.477681, abs=1e-5)
    assert e["27L"].opposite == "09R" and e["27R"].opposite == "09L"
    assert e["09R"].opposite == "27L" and e["09L"].opposite == "27R"
    for r in e.values():
        assert r.heading == pytest.approx(90 if r.ident.startswith("09") else 270, abs=1.0)
        # runways.csv lists 12,799 ft (09L/27R) and 12,001 ft (09R/27L)
        assert r.length_m == pytest.approx(3_901 if r.ident in ("09L", "27R") else 3_658, abs=60)


def test_displaced_threshold_moves_landing_point_inboard(ref):
    e = {r.ident: r for r in ref.airports["EGLL"].runways}
    # 09R lists a 1,013 ft displaced threshold: landing threshold is ~309 m east of the pavement end.
    assert e["09R"].displaced_m == pytest.approx(1013 * 0.3048)
    assert e["09R"].thr_lon > e["09R"].lon
    # 27L lists none: threshold is the pavement end.
    assert (e["27L"].thr_lat, e["27L"].thr_lon) == (e["27L"].lat, e["27L"].lon)


def test_dft_join_and_filters(ref):
    assert ref.airports["EGLL"].dft_atm == 473_799
    assert ref.airports["EGLL"].dft_year == 2024
    assert ref.airports["EGLL"].iata == "LHR"
    assert "EGFF" in ref.airports  # Wales
    assert "EGJJ" not in ref.airports  # Jersey is not GB, and not in AVI0102
    assert set(ref.airports) == {"EGLL", "EGKK", "EGCC", "EGFF"}


def test_ends_near_and_lookup(ref):
    r = ref.runway_end("EGLL", "27L")
    assert r is not None
    assert r in ref.ends_near(r.thr_lat, r.thr_lon, 100)
    assert ref.runway_end("EGLL", "99") is None
    assert ref.ends_near(55.0, -3.0, 5000) == []


def test_content_hash_is_deterministic_and_sensitive(ref):
    again = build_reference(FIX / "airports_trim.csv", FIX / "runways_trim.csv", DEFAULT_DFT_PATH)
    assert ref.content_hash() == again.content_hash()
    again.airports["EGLL"].dft_atm = 1
    assert ref.content_hash() != again.content_hash()


def test_skips_closed_water_and_helipad_ends(tmp_path):
    ap = tmp_path / "a.csv"
    rw = tmp_path / "r.csv"
    ap.write_text(
        'id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,icao_code,iata_code,gps_code,local_code,home_link,wikipedia_link,keywords\n'
        '1,EGXX,medium_airport,Test,52.0,-1.0,100,EU,GB,GB-ENG,x,yes,EGXX,TST,EGXX,,,,\n'
        '2,EGYY,small_airport,No Iata,52.5,-1.0,100,EU,GB,GB-ENG,x,no,EGYY,,EGYY,,,,\n'
    )
    head = "id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,le_ident,le_latitude_deg,le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,he_ident,he_latitude_deg,he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft\n"
    rw.write_text(
        head
        + '10,1,EGXX,5000,100,ASP,1,1,09,52.0,-1.01,,90,,27,52.0,-0.99,,270,\n'  # closed
        + '11,1,EGXX,5000,100,WATER,1,0,18,52.0,-1.0,,180,,36,52.0,-1.0,,0,\n'  # water
        + '12,1,EGXX,100,100,ASP,1,0,H1,52.0,-1.0,,,,,,,,,\n'  # helipad
        + '13,2,EGYY,5000,100,ASP,1,0,09,52.5,-1.01,,90,,27,52.5,-0.99,,270,\n'
    )
    r = build_reference(ap, rw, None)
    assert r.airports == {}  # EGXX has no usable ends; EGYY is small without IATA
    r2 = build_reference(ap, rw, None, include_small_without_iata=True)
    assert set(r2.airports) == {"EGYY"}
    assert {e.ident for e in r2.airports["EGYY"].runways} == {"09", "27"}
