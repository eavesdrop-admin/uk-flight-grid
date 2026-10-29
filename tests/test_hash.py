from __future__ import annotations

import hashlib

import extract_day as ex
from trace_builders import pt, trace


def test_uid_is_first_eight_bytes_of_salted_sha256():
    salt = b"s" * 32
    expect = int.from_bytes(hashlib.sha256(salt + b"4ca7b2").digest()[:8], "big")
    assert ex.daily_uid(salt, "4ca7b2") == expect
    assert 0 <= expect < 2**64


def test_same_aircraft_gets_different_uids_on_different_days():
    a = ex.daily_uid(b"day-one-salt".ljust(32, b"."), "4ca7b2")
    b = ex.daily_uid(b"day-two-salt".ljust(32, b"."), "4ca7b2")
    assert a != b


def test_uid_is_stable_within_a_day_and_distinct_between_aircraft(salt):
    assert ex.daily_uid(salt, "4ca7b2") == ex.daily_uid(salt, "4ca7b2")
    assert ex.daily_uid(salt, "4ca7b2") != ex.daily_uid(salt, "4ca7b3")


def test_uid_does_not_leak_the_icao_address(salt):
    assert ex.daily_uid(salt, "4ca7b2") != int("4ca7b2", 16)


def test_trace_rows_carry_the_hashed_uid(salt):
    cols, _ = ex.extract_trace(trace([pt(0), pt(60)], icao="4ca7b2"), salt)
    assert cols["uid"] == [ex.daily_uid(salt, "4ca7b2")] * 2


def test_no_collisions_across_a_realistic_day(salt):
    uids = {ex.daily_uid(salt, f"{i:06x}") for i in range(0, 0x1000000, 331)}  # ~50k addresses
    assert len(uids) == len(range(0, 0x1000000, 331))
