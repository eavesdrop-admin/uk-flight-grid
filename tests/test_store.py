import json

import pytest
from botocore.exceptions import ClientError

import store
from merge_manifests import merge


class FakeS3:
    def __init__(self, keys):
        self.keys = dict(keys)
        self.puts = []

    def head_object(self, Bucket, Key):
        if Key not in self.keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def download_file(self, bucket, key, dest):
        open(dest, "w").write(self.keys[key])

    def upload_file(self, local, bucket, key):
        self.puts.append(key)


def test_missing_env_names_the_secrets():
    assert store.missing_env({}) == list(store.REQUIRED_ENV)
    full = {k: "x" for k in store.REQUIRED_ENV}
    assert store.missing_env(full) == []


def test_day_keys_layout():
    assert store.day_keys("2026-09-05") == ("extract/2026/2026-09-05.parquet", "extract/manifests/2026-09-05.json")


def test_exists_distinguishes_missing_from_error():
    s3 = FakeS3({"a": ""})
    assert store.exists(s3, "a") and not store.exists(s3, "b")

    class Boom(FakeS3):
        def head_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "403"}}, "HeadObject")

    with pytest.raises(ClientError):
        store.exists(Boom({}), "a")


def test_get_days_downloads_present_days_and_reports_missing(tmp_path):
    s3 = FakeS3({"extract/2026/2026-09-01.parquet": "pq", "extract/manifests/2026-09-01.json": "{}"})
    got, missing = store.get_days(s3, tmp_path, ["2026-09-01", "2026-09-05"])
    assert got == ["2026-09-01"] and missing == ["2026-09-05"]
    assert (tmp_path / "2026" / "2026-09-01.parquet").read_text() == "pq"
    assert (tmp_path / "manifests" / "2026-09-01.json").exists()


def test_put_dir_uploads_files_under_the_prefix(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "cells.parquet").write_text("x")
    (tmp_path / "sub").mkdir()
    s3 = FakeS3({})
    assert store.put_dir(s3, tmp_path, "grid/2026-09-30/") == ["grid/2026-09-30/cells.parquet", "grid/2026-09-30/meta.json"]


def test_merge_manifests_re_evaluates_degraded_days_across_all_days(tmp_path):
    for i in range(1, 9):  # 8 normal days then one with under 60% of the median
        day = f"2026-09-{i:02d}"
        (tmp_path / f"{day}.json").write_text(json.dumps({"days": {day: {"points_kept": 1000, "status": "ok"}}}))
    (tmp_path / "2026-09-09.json").write_text(json.dumps({"days": {"2026-09-09": {"points_kept": 300, "status": "ok"}}}))
    m = merge(tmp_path)
    assert m["days"]["2026-09-09"]["status"] == "degraded"
    assert m["days"]["2026-09-08"]["status"] == "ok"
