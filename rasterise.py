"""Rasterise resampled legs onto H3 cells.

For each leg we reduce its samples to the unique res-9 cells it passes through (keeping the
lowest altitude and the time of that lowest sample), then expand to the "overhead" set
(grid_disk k=1, about 0.5 km) and the "nearby" set (grid_disk k=6, about 2 km). Within a leg
every cell appears once, with the lowest altitude of any path cell that reaches it, so a leg
is never counted twice in a cell however many samples fall near it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from h3.api import basic_int as h3i

from grid_schema import CELL_RES, MAX_COUNTED_ALT_FT, NEARBY_K, OVERHEAD_K
from segment import Sample


@dataclass
class CellSet:
    """One leg's cells: unique cells with the lowest altitude, its time and an inferred flag."""

    cells: np.ndarray  # uint64
    alt_ft: np.ndarray  # float32, lowest altitude of any contributing sample
    t: np.ndarray  # float64, epoch seconds of that sample
    inferred: np.ndarray  # bool, True when every contributing sample is synthetic

    def __len__(self) -> int:
        return len(self.cells)


@dataclass
class LegRaster:
    overhead: CellSet
    nearby: CellSet


def _empty() -> CellSet:
    return CellSet(
        np.empty(0, np.uint64), np.empty(0, np.float32), np.empty(0, np.float64), np.empty(0, bool)
    )


@lru_cache(maxsize=400_000)
def _disk1(cell: int) -> np.ndarray:
    return np.fromiter(h3i.grid_disk(cell, 1), dtype=np.uint64)


def neighbours(cell: int, k: int) -> np.ndarray:
    """Cells within k steps of `cell`, including itself (1, 7, 19, ... 127 for k=6)."""
    if k == 1:
        return _disk1(cell)
    return np.fromiter(h3i.grid_disk(cell, k), dtype=np.uint64)


def path_cells(
    lat: np.ndarray,
    lon: np.ndarray,
    alt_ft: np.ndarray,
    t: np.ndarray,
    inferred: np.ndarray | None = None,
    res: int = CELL_RES,
    max_alt_ft: float = MAX_COUNTED_ALT_FT,
) -> CellSet:
    """Unique res-`res` cells of the samples below max_alt_ft, lowest altitude per cell."""
    lat, lon, alt_ft, t = (np.asarray(a) for a in (lat, lon, alt_ft, t))
    inf = np.zeros(len(lat), bool) if inferred is None else np.asarray(inferred, bool)
    keep = alt_ft < max_alt_ft
    if not keep.any():
        return _empty()
    lat, lon, alt_ft, t, inf = lat[keep], lon[keep], alt_ft[keep], t[keep], inf[keep]
    cells = np.fromiter(
        (h3i.latlng_to_cell(a, b, res) for a, b in zip(lat.tolist(), lon.tolist())),
        dtype=np.uint64,
        count=len(lat),
    )
    order = np.lexsort((alt_ft, cells))  # by cell, then lowest altitude first
    c, a, tt, ii = cells[order], alt_ft[order], t[order], inf[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return CellSet(
        c[starts], a[starts].astype(np.float32), tt[starts].astype(np.float64),
        np.logical_and.reduceat(ii, starts),
    )


def expand(path: CellSet, k: int) -> CellSet:
    """Every cell within k steps of any path cell, lowest altitude (and its time) per cell."""
    if len(path) == 0:
        return _empty()
    if k == 0:
        return path
    nbrs = [neighbours(int(c), k) for c in path.cells]
    lens = np.fromiter((len(n) for n in nbrs), dtype=np.int64, count=len(nbrs))
    flat = np.concatenate(nbrs)
    idx = np.repeat(np.arange(len(nbrs)), lens)
    alt, t, inf = path.alt_ft[idx], path.t[idx], path.inferred[idx]
    order = np.lexsort((alt, flat))
    f = flat[order]
    starts = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    return CellSet(
        f[starts], alt[order][starts], t[order][starts],
        np.logical_and.reduceat(inf[order], starts),
    )


def rasterise_samples(
    samples: Sequence[Sample],
    inferred_tail: Sequence[Sample] = (),
    fallback_t: float = 0.0,
    overhead_k: int = OVERHEAD_K,
    nearby_k: int = NEARBY_K,
) -> LegRaster:
    """Rasterise one leg's observed/interpolated samples plus any synthetic final approach.

    Synthetic tail samples have no time; they take `fallback_t` (the leg's last time).
    """
    allsamp = list(samples) + list(inferred_tail)
    if not allsamp:
        return LegRaster(_empty(), _empty())
    n_obs = len(samples)
    lat = np.array([s.lat for s in allsamp])
    lon = np.array([s.lon for s in allsamp])
    alt = np.array([s.alt_ft for s in allsamp], dtype=np.float64)
    t = np.array([s.t if i < n_obs else fallback_t for i, s in enumerate(allsamp)])
    inf = np.arange(len(allsamp)) >= n_obs
    path = path_cells(lat, lon, alt, t, inf)
    return LegRaster(expand(path, overhead_k), expand(path, nearby_k))
