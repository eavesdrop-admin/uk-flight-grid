from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from aggregate import ALT_FLOOR_FT, DayAgg, aggregate_day, alt_bin, band_index, local_hour, merge_days, night_mask
from grid_schema import CELLS_SCHEMA


def utc(*a) -> float:
    return datetime(*a, tzinfo=timezone.utc).timestamp()


# ------------------------------------------------------- night / DST


def test_night_is_2300_to_0700_local_in_winter():
    # 2026-01-15 is GMT: local == UTC
    ts = [utc(2026, 1, 15, h, m) for h, m in [(22, 59), (23, 0), (3, 0), (6, 59), (7, 0)]]
    assert night_mask(np.array(ts)).tolist() == [False, True, True, True, False]


def test_night_is_2300_to_0700_local_in_summer():
    # 2026-07-15 is BST: local == UTC + 1
    ts = [utc(2026, 7, 15, h, m) for h, m in [(21, 59), (22, 0), (5, 59), (6, 0)]]
    assert night_mask(np.array(ts)).tolist() == [False, True, True, False]


def test_night_boundary_across_the_march_clock_change():
    # Clocks go forward at 01:00 UTC on Sunday 2026-03-29.
    assert local_hour(np.array([utc(2026, 3, 28, 22, 30)])).tolist() == [22]  # GMT
    assert local_hour(np.array([utc(2026, 3, 29, 22, 30)])).tolist() == [23]  # BST
    assert night_mask(np.array([utc(2026, 3, 28, 22, 30)])).tolist() == [False]
    assert night_mask(np.array([utc(2026, 3, 29, 22, 30)])).tolist() == [True]
    # morning boundary: 06:30 UTC is 06:30 GMT (night) the day before, 07:30 BST (day) after
    assert night_mask(np.array([utc(2026, 3, 28, 6, 30)])).tolist() == [True]
    assert night_mask(np.array([utc(2026, 3, 29, 6, 30)])).tolist() == [False]
    # the instant the clocks change: 00:59:59 UTC is 00:59:59 GMT, 01:00:00 UTC is 02:00:00 BST; both night
    assert night_mask(np.array([utc(2026, 3, 29, 0, 59, 59), utc(2026, 3, 29, 1, 0, 0)])).tolist() == [True, True]


def test_night_boundary_across_the_october_clock_change():
    # Clocks go back at 01:00 UTC on Sunday 2026-10-25.
    assert night_mask(np.array([utc(2026, 10, 24, 6, 30)])).tolist() == [False]  # 07:30 BST
    assert night_mask(np.array([utc(2026, 10, 25, 6, 30)])).tolist() == [True]  # 06:30 GMT
    assert night_mask(np.array([utc(2026, 10, 24, 22, 30)])).tolist() == [True]  # 23:30 BST
    assert night_mask(np.array([utc(2026, 10, 25, 22, 30)])).tolist() == [False]  # 22:30 GMT
    assert night_mask(np.array([utc(2026, 10, 25, 22, 59)])).tolist() == [False]
    assert night_mask(np.array([utc(2026, 10, 25, 23, 0)])).tolist() == [True]  # 23:00 GMT
    # 01:30 local happens twice that morning (00:30 UTC BST and 01:30 UTC GMT); both are night
    assert night_mask(np.array([utc(2026, 10, 25, 0, 30), utc(2026, 10, 25, 1, 30)])).tolist() == [True, True]


def test_band_index_edges():
    assert band_index(np.array([-50.0, 0.0, 999.9, 1000.0, 2999.9, 3000.0, 6999.9])).tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_alt_bin_range():
    assert alt_bin(np.array([ALT_FLOOR_FT, 0.0, 6999.0, 20_000.0])).tolist() == [0, 10, 79, 79]


# ------------------------------------------------------- one day


def leg_cells(rows, with_inferred):
    """rows: (leg_id, cell, alt_ft, t_epoch[, inferred])"""
    cols = ["leg_id", "cell", "alt_ft", "t"] + (["inferred"] if with_inferred else [])
    df = pl.DataFrame(rows, schema=cols, orient="row")
    return df.cast({"leg_id": pl.UInt32, "cell": pl.UInt64, "alt_ft": pl.Float32})


def legs_table(rows):
    return pl.DataFrame(rows, schema={"leg_id": pl.UInt32, "airport": pl.String, "runway": pl.String, "op": pl.String}, orient="row")


DAY = utc(2026, 1, 15, 12)  # midday, GMT
NIGHT = utc(2026, 1, 15, 2)


def one_day(oh_rows, nb_rows, legs):
    return aggregate_day(leg_cells(oh_rows, True), leg_cells(nb_rows, False), legs_table(legs))


def test_aggregate_day_counts_bands_night_and_inferred():
    # cell 100: leg0 at 500 ft midday, leg1 at 2000 ft at night (inferred), leg2 at 5000 ft midday
    oh = [(0, 100, 500, DAY, False), (1, 100, 2000, NIGHT, True), (2, 100, 5000, DAY, False)]
    nb = [(0, 100, 500, DAY), (1, 100, 2000, NIGHT), (2, 100, 5000, DAY), (0, 101, 500, DAY)]
    agg = one_day(oh, nb, [(0, "EGLL", "27L", "A"), (1, "EGLL", "27L", "A"), (2, "", "", "O")])
    c = agg.cells.filter(pl.col("cell") == 100).row(0, named=True)
    assert (c["oh_n"], c["oh_b0"], c["oh_b1"], c["oh_b2"], c["oh_night"], c["oh_inf"]) == (3, 1, 1, 1, 1, 1)
    assert (c["nb_n"], c["nb_night"]) == (3, 1)
    # cell 101 is only nearby
    c101 = agg.cells.filter(pl.col("cell") == 101).row(0, named=True)
    assert (c101["oh_n"], c101["nb_n"]) == (0, 1)
    att = {(r["airport"], r["runway"], r["op"]): r["n"] for r in agg.attrib.filter(pl.col("cell") == 100).iter_rows(named=True)}
    assert att == {("EGLL", "27L", "A"): 2, ("", "", "O"): 1}
    assert agg.alt_hist["n"].sum() == 3


# ------------------------------------------------------- merge


def day(oh_counts, nb_counts=None, *, alt=1200, att=("EGLL", "27L", "A"), night=0):
    """A DayAgg with one overhead cell (id 7) seeing `oh_counts` legs, all in one band."""
    nb_counts = oh_counts if nb_counts is None else nb_counts
    band = 0 if alt < 1000 else 1 if alt < 3000 else 2
    cells = pl.DataFrame(
        {
            "cell": [7],
            "nb_n": [nb_counts], "nb_b0": [nb_counts if band == 0 else 0], "nb_b1": [nb_counts if band == 1 else 0],
            "nb_b2": [nb_counts if band == 2 else 0], "nb_night": [0],
            "oh_n": [oh_counts], "oh_b0": [oh_counts if band == 0 else 0], "oh_b1": [oh_counts if band == 1 else 0],
            "oh_b2": [oh_counts if band == 2 else 0], "oh_night": [night], "oh_inf": [0],
        }
    ).cast(pl.UInt32).with_columns(pl.col("cell").cast(pl.UInt64))
    from aggregate import ALT_BIN_FT

    hist = pl.DataFrame({"cell": [7], "bin": [int((alt - ALT_FLOOR_FT) // ALT_BIN_FT)], "n": [oh_counts]}).cast(
        {"cell": pl.UInt64, "bin": pl.UInt8, "n": pl.UInt32}
    )
    attrib = pl.DataFrame({"cell": [7], "airport": [att[0]], "runway": [att[1]], "op": [att[2]], "n": [oh_counts]}).cast(
        {"cell": pl.UInt64, "n": pl.UInt32}
    )
    if oh_counts == 0:
        hist, attrib = hist.clear(), attrib.clear()
        if nb_counts == 0:
            cells = cells.clear()
    return DayAgg(cells, hist, attrib)


def test_mean_and_p90_include_zero_days():
    # Four observed days; the cell is only seen on the first, with 4 legs.
    days = [day(4)] + [day(0, 0)] * 3
    out = merge_days(days)
    r = out.row(0, named=True)
    assert r["oh_day_mean"] == pytest.approx(1.0)  # 4 / 4 days, not 4 / 1
    assert r["oh_days_active"] == 1
    # numpy percentile of [4,0,0,0] at 90: sorted [0,0,0,4], position 2.7 -> 2.8, rounds to 3
    assert r["oh_day_p90"] == 3
    # excluding the zero days would have said 4: that is the bug this guards against
    assert r["oh_day_p90"] != 4


def test_p90_of_steady_traffic():
    out = merge_days([day(c) for c in [10, 10, 10, 10, 20]])
    r = out.row(0, named=True)
    assert r["oh_day_mean"] == pytest.approx(12.0)
    assert r["oh_day_p90"] == 16  # sorted [10,10,10,10,20], position 0.9 * 4 = 3.6 -> 10 + 0.6 * 10


def test_bands_are_means_over_all_days():
    out = merge_days([day(6, alt=500), day(6, alt=2000), day(0, 0)])
    r = out.row(0, named=True)
    assert r["oh_lt1000"] == pytest.approx(2.0) and r["oh_1000_3000"] == pytest.approx(2.0) and r["oh_3000_7000"] == 0
    assert r["oh_day_mean"] == pytest.approx(4.0)
    assert r["oh_lt1000"] + r["oh_1000_3000"] + r["oh_3000_7000"] == pytest.approx(r["oh_day_mean"])


def test_night_mean():
    out = merge_days([day(10, night=4), day(10, night=2)])
    assert out["oh_night_mean"][0] == pytest.approx(3.0)


def test_sparse_filter_keeps_cells_at_point_one_per_day_or_more():
    keep = merge_days([day(1)] + [day(0, 0)] * 9)  # 1 leg in 10 days = 0.1/day: kept
    drop = merge_days([day(1)] + [day(0, 0)] * 10)  # 1 in 11 days: below 0.1/day, dropped
    assert keep.height == 1 and drop.height == 0


def test_nearby_only_cell_is_kept_with_null_overhead_fields():
    out = merge_days([day(0, 5)])
    r = out.row(0, named=True)
    assert r["oh_day_mean"] == 0 and r["nb_day_mean"] == 5.0
    assert r["oh_alt_p50_ft"] is None and r["top1_op"] is None and r["top1_share"] is None


def test_median_altitude_is_bin_centre():
    out = merge_days([day(5, alt=1230)])
    assert out["oh_alt_p50_ft"][0] == 1250  # 100 ft bins labelled by centre


def test_top_two_and_shares():
    a = day(6, att=("EGLL", "27L", "A"))
    b = day(3, att=("EGLL", "27R", "A"))
    c = day(1, att=("", "", "O"))
    # three different attributions in the same cell: stack the per-day frames
    cells = a.cells.with_columns(
        *[pl.lit(10, dtype=pl.UInt32).alias(k) for k in ("oh_n", "nb_n", "oh_b1", "nb_b1")]
    )
    merged = DayAgg(cells, a.alt_hist.with_columns(pl.lit(10, dtype=pl.UInt32).alias("n")), pl.concat([a.attrib, b.attrib, c.attrib]))
    r = merge_days([merged]).row(0, named=True)
    assert (r["top1_airport"], r["top1_runway"], r["top1_op"], r["top1_share"]) == ("EGLL", "27L", "A", 60)
    assert (r["top2_airport"], r["top2_runway"], r["top2_op"], r["top2_share"]) == ("EGLL", "27R", "A", 30)


def test_overflight_attribution_has_null_airport_and_runway():
    r = merge_days([day(5, att=("", "", "O"))]).row(0, named=True)
    assert r["top1_airport"] is None and r["top1_runway"] is None and r["top1_op"] == "O" and r["top1_share"] == 100


def test_output_matches_the_schema_exactly():
    out = merge_days([day(3), day(2)])
    assert dict(out.schema) == CELLS_SCHEMA
    assert out.columns == list(CELLS_SCHEMA)


def test_no_days_is_an_error():
    with pytest.raises(ValueError):
        merge_days([])
