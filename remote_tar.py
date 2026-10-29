"""Seekable, resumable view over the concatenated parts of a split tar.

adsb.lol publishes each day as ``<tag>.tar.aa``, ``.tar.ab`` ... (plain, uncompressed
tar cut into 2 GB pieces). This presents the parts as one read-only file so that
``tarfile`` can open it in ``"r:"`` mode. Forward seeks past a few MiB are done by
dropping the connection and issuing an HTTP Range request, so entries we do not
want (``heatmap/``, about a quarter of the day) are never downloaded.

Parts may be http(s) URLs or local paths (used by tests and for re-running from a
local copy).
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import BinaryIO

USER_AGENT = "uk-flight-grid (+https://github.com/adsblol/globe_history_2026)"

# A forward seek shorter than this is cheaper to read-and-discard than to reconnect.
SKIP_READ_LIMIT = 8 * 1024 * 1024
CONNECT_TIMEOUT_S = 60
MAX_RETRIES = 6

Opener = Callable[[str, int], BinaryIO]


def _is_url(location: str) -> bool:
    return location.startswith(("http://", "https://"))


def default_opener(location: str, offset: int) -> BinaryIO:
    """Open ``location`` positioned at byte ``offset`` (HTTP Range for URLs)."""
    if _is_url(location):
        req = urllib.request.Request(
            location,
            headers={"Range": f"bytes={offset}-", "User-Agent": USER_AGENT},
        )
        resp = urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S)
        status = getattr(resp, "status", 206)
        if offset and status != 206:
            resp.close()
            raise OSError(f"server ignored Range request (HTTP {status}) for {location}")
        return resp  # type: ignore[return-value]
    fh = open(location, "rb")
    fh.seek(offset)
    return fh


def url_size(location: str) -> int:
    """Size in bytes of a part (HEAD for URLs, stat for paths)."""
    if not _is_url(location):
        import os

        return os.path.getsize(location)
    req = urllib.request.Request(location, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
        return int(resp.headers["Content-Length"])


class SplitTarFile(io.RawIOBase):
    """Read-only, seekable file over ``parts`` laid end to end.

    ``parts`` is a sequence of ``(location, size)``. Sizes must be exact (taken from the
    release asset metadata), because they define where one part ends and the next begins.
    """

    def __init__(
        self,
        parts: Sequence[tuple[str, int]],
        opener: Opener = default_opener,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__()
        self._parts = list(parts)
        self._opener = opener
        self._sleep = sleep
        self._starts: list[int] = []
        total = 0
        for _, size in self._parts:
            self._starts.append(total)
            total += size
        self._size = total
        self._pos = 0
        self._conn: BinaryIO | None = None
        self._conn_part = -1
        self._conn_pos = 0  # absolute position the open connection will read next
        self.bytes_fetched = 0  # bytes actually pulled over the wire / from disk

    # -- io.RawIOBase interface -------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return new

    def readinto(self, b) -> int:  # type: ignore[override]
        view = memoryview(b).cast("B")
        want = len(view)
        got = 0
        while got < want and self._pos < self._size:
            part = self._part_index(self._pos)
            part_end = self._starts[part] + self._parts[part][1]
            take = min(want - got, part_end - self._pos)
            chunk = self._read_from_connection(part, take)
            if not chunk:
                raise EOFError(f"unexpected end of part {part} at {self._pos}")
            view[got : got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got

    def close(self) -> None:
        self._drop_connection()
        super().close()

    # -- internals --------------------------------------------------------------
    def _part_index(self, pos: int) -> int:
        for i in range(len(self._parts) - 1, -1, -1):
            if pos >= self._starts[i]:
                return i
        return 0

    def _drop_connection(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._conn_part = -1

    def _connect(self, part: int, pos: int) -> None:
        self._drop_connection()
        location = self._parts[part][0]
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._conn = self._opener(location, pos - self._starts[part])
                self._conn_part = part
                self._conn_pos = pos
                return
            except (OSError, urllib.error.URLError) as exc:
                last = exc
                self._sleep(min(2**attempt, 30))
        raise OSError(f"could not open {location} at offset {pos - self._starts[part]}") from last

    def _read_from_connection(self, part: int, n: int) -> bytes:
        pos = self._pos
        for attempt in range(MAX_RETRIES):
            try:
                if self._conn is None or self._conn_part != part or self._conn_pos > pos:
                    self._connect(part, pos)
                elif self._conn_pos < pos:
                    gap = pos - self._conn_pos
                    if gap > SKIP_READ_LIMIT:
                        self._connect(part, pos)
                    else:
                        self._discard(gap)
                assert self._conn is not None
                data = self._conn.read(n)
                self._conn_pos += len(data)
                self.bytes_fetched += len(data)
                return data
            except (OSError, urllib.error.URLError, EOFError):
                # Resume from the byte we still need.
                self._drop_connection()
                if attempt == MAX_RETRIES - 1:
                    raise
                self._sleep(min(2**attempt, 30))
        raise AssertionError("unreachable")

    def _discard(self, n: int) -> None:
        assert self._conn is not None
        left = n
        while left > 0:
            data = self._conn.read(min(left, 1024 * 1024))
            if not data:
                raise EOFError("connection ended while skipping")
            left -= len(data)
            self._conn_pos += len(data)
            self.bytes_fetched += len(data)
