import numpy as np
from h3.api import basic_int as h3i

from grid_schema import NEARBY_K, OVERHEAD_K
from rasterise import expand, neighbours, path_cells, rasterise_samples
from segment import Sample

LAT, LON = 51.47, -0.36


def test_neighbour_set_sizes():
    c = h3i.latlng_to_cell(LAT, LON, 9)
    assert len(neighbours(c, 1)) == 7
    assert len(neighbours(c, 6)) == 127
    assert set(neighbours(c, 1).tolist()) <= set(neighbours(c, 6).tolist())
    assert c in neighbours(c, 1).tolist()


def test_configured_radii_match_the_plan():
    assert (OVERHEAD_K, NEARBY_K) == (1, 6)
    # res 9 edge is about 174 m: k=1 reaches about 0.5 km, k=6 about 2 km
    c = h3i.latlng_to_cell(LAT, LON, 9)
    far = max(h3i.grid_distance(c, n) for n in neighbours(c, 6).tolist())
    assert far == 6


def test_path_cells_dedupes_and_keeps_lowest_altitude():
    lat = np.array([LAT, LAT, LAT])
    lon = np.array([LON, LON, LON])
    ps = path_cells(lat, lon, np.array([2500.0, 900.0, 1500.0]), np.array([10.0, 20.0, 30.0]))
    assert len(ps) == 1
    assert ps.alt_ft[0] == 900.0 and ps.t[0] == 20.0


def test_path_cells_drops_samples_at_or_above_7000():
    ps = path_cells(np.array([LAT, LAT + 0.1]), np.array([LON, LON]), np.array([7000.0, 6999.0]), np.array([0.0, 1.0]))
    assert len(ps) == 1 and ps.alt_ft[0] == 6999.0
    assert len(path_cells(np.array([LAT]), np.array([LON]), np.array([9000.0]), np.array([0.0]))) == 0


def test_inferred_only_when_every_sample_in_the_cell_is_synthetic():
    lat, lon, t = np.array([LAT, LAT]), np.array([LON, LON]), np.array([0.0, 1.0])
    alt = np.array([500.0, 600.0])
    assert path_cells(lat, lon, alt, t, np.array([True, True])).inferred[0]
    assert not path_cells(lat, lon, alt, t, np.array([True, False])).inferred[0]


def test_expand_counts_each_cell_once_with_lowest_altitude_of_any_path_cell():
    a = h3i.latlng_to_cell(LAT, LON, 9)
    b = next(n for n in neighbours(a, 1).tolist() if n != a)  # adjacent path cell
    from rasterise import CellSet

    path = CellSet(
        np.array([a, b], dtype=np.uint64), np.array([2500.0, 500.0], np.float32),
        np.array([1.0, 2.0]), np.array([False, False]),
    )
    out = expand(path, 1)
    assert len(set(out.cells.tolist())) == len(out.cells)
    assert set(out.cells.tolist()) == set(neighbours(a, 1).tolist()) | set(neighbours(b, 1).tolist())
    shared = set(neighbours(a, 1).tolist()) & set(neighbours(b, 1).tolist())
    for cell, alt, t in zip(out.cells.tolist(), out.alt_ft, out.t):
        if cell in shared:
            assert alt == 500.0 and t == 2.0  # reached by both: the lower pass wins
    assert out.alt_ft[out.cells.tolist().index(a)] == 500.0  # a is within 1 step of b


def test_rasterise_samples_overhead_is_subset_of_nearby_and_unique():
    samples = [Sample(float(i), LAT, LON + i * 0.0015, 1200.0, False) for i in range(40)]
    r = rasterise_samples(samples)
    oh, nb = set(r.overhead.cells.tolist()), set(r.nearby.cells.tolist())
    assert oh < nb
    assert len(oh) == len(r.overhead) and len(nb) == len(r.nearby)
    assert len(nb) > 4 * len(oh)  # the 2 km set is much wider than the 0.5 km set


def test_synthetic_tail_is_flagged_inferred_and_takes_the_fallback_time():
    obs = [Sample(float(i), LAT, LON + i * 0.001, 1500.0, False) for i in range(10)]
    tail = [Sample(float("nan"), LAT, LON + 0.02 + i * 0.001, 1000.0, True) for i in range(10)]
    r = rasterise_samples(obs, tail, fallback_t=99.0)
    assert r.overhead.inferred.any() and not r.overhead.inferred.all()
    assert not np.isnan(r.overhead.t).any()
    assert 99.0 in r.overhead.t
