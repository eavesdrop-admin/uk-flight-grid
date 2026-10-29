from __future__ import annotations

import io
import tarfile
import urllib.error

import pytest

import remote_tar as rt


def make_tar(path_prefix, members: dict[str, bytes], parts: int):
    """Write a tar and cut it into ``parts`` pieces; returns [(path, size)]."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    cut = -(-len(blob) // parts)
    out = []
    for i in range(parts):
        p = f"{path_prefix}.{chr(97)}{chr(97 + i)}"
        chunk = blob[i * cut : (i + 1) * cut]
        with open(p, "wb") as fh:
            fh.write(chunk)
        out.append((p, len(chunk)))
    return out, blob


def test_reads_across_part_boundaries_and_seeks(tmp_path):
    blob = bytes(range(256)) * 40
    parts = []
    for i in range(3):
        p = tmp_path / f"p{i}"
        chunk = blob[i * 3500 : (i + 1) * 3500]
        p.write_bytes(chunk)
        parts.append((str(p), len(chunk)))
    f = rt.SplitTarFile(parts)
    assert f.read(10) == blob[:10]
    f.seek(3495)
    assert f.read(10) == blob[3495:3505]  # crosses part 0 -> 1
    f.seek(-5, 2)
    assert f.read() == blob[-5:]
    f.seek(100)
    assert f.read(7000) == blob[100:7100]
    assert f.read() == blob[7100:]


def test_tarfile_opens_split_parts_and_skips_unwanted_members_without_reading_them(tmp_path):
    members = {
        "./heatmap/0.bin": b"H" * 3_000_000,
        "./traces/00/a.json": b"one",
        "./traces/01/b.json": b"two",
    }
    parts, _ = make_tar(str(tmp_path / "day.tar"), members, parts=2)
    f = rt.SplitTarFile(parts)
    with tarfile.open(fileobj=f, mode="r:") as tf:
        names = [m.name for m in tf]
        assert names == list(members)
    f.seek(0)
    got = {}
    with tarfile.open(fileobj=f, mode="r:") as tf:
        for m in tf:
            if m.name.startswith("./traces/"):
                got[m.name] = tf.extractfile(m).read()
    assert got == {"./traces/00/a.json": b"one", "./traces/01/b.json": b"two"}


def test_long_forward_seek_reconnects_instead_of_downloading(tmp_path):
    blob = (b"x" * (rt.SKIP_READ_LIMIT * 3)) + b"TAIL"
    p = tmp_path / "one"
    p.write_bytes(blob)
    opened: list[int] = []

    def opener(loc, off):
        opened.append(off)
        return rt.default_opener(loc, off)

    f = rt.SplitTarFile([(str(p), len(blob))], opener=opener)
    f.read(10)
    f.seek(len(blob) - 4)
    assert f.read(4) == b"TAIL"
    assert opened == [0, len(blob) - 4]
    assert f.bytes_fetched < 100  # nothing like the 24 MB that was skipped


def test_short_forward_skip_reuses_the_connection(tmp_path):
    p = tmp_path / "one"
    p.write_bytes(bytes(range(256)) * 100)
    opened: list[int] = []

    def opener(loc, off):
        opened.append(off)
        return rt.default_opener(loc, off)

    f = rt.SplitTarFile([(str(p), 25600)], opener=opener)
    f.read(10)
    f.seek(1000)
    f.read(10)
    assert opened == [0]


def test_backwards_seek_reopens(tmp_path):
    p = tmp_path / "one"
    p.write_bytes(bytes(range(200)))
    f = rt.SplitTarFile([(str(p), 200)])
    f.read(100)
    f.seek(5)
    assert f.read(3) == bytes([5, 6, 7])


class FlakyStream(io.BytesIO):
    """Raises after ``fail_after`` bytes, like a dropped connection."""

    def __init__(self, data, fail_after):
        super().__init__(data)
        self.fail_after = fail_after
        self.served = 0

    def read(self, n=-1):
        if self.served >= self.fail_after:
            raise ConnectionResetError("boom")
        data = super().read(min(n, self.fail_after - self.served) if n >= 0 else -1)
        self.served += len(data)
        return data


def test_dropped_connection_resumes_from_the_right_byte():
    blob = bytes(range(256)) * 20
    opens: list[int] = []

    def opener(loc, off):
        opens.append(off)
        # the first connection dies after 1000 bytes; later ones are healthy
        return FlakyStream(blob[off:], 1000 if len(opens) == 1 else 10**9)

    f = rt.SplitTarFile([("mem://x", len(blob))], opener=opener, sleep=lambda s: None)
    assert f.read(3000) == blob[:3000]
    assert opens == [0, 1000]


def test_gives_up_after_repeated_failures():
    def opener(loc, off):
        raise urllib.error.URLError("down")

    f = rt.SplitTarFile([("mem://x", 10)], opener=opener, sleep=lambda s: None)
    with pytest.raises(OSError):
        f.read(1)
