"""Private object storage for extracts and grids (S3 API, Supabase Storage in practice).

Everything this repo derives from the adsb.lol archive goes here and nowhere public: no
Actions artifacts, no release assets. Layout in the bucket:

    extract/<YYYY>/<date>.parquet        Stage A output, one per day
    extract/manifests/<date>.json        that day's manifest entry
    grid/<window_end>/...                a build that passed validation (the Eavesdrop loader reads the newest)
    grid-failed/<window_end>/...         a build that did not; never read by the loader
    pilot/<YYYY-MM>/sampling_check.json  the sampled-vs-full pilot result

Environment (the same names the Eavesdrop loader uses):
    FLIGHT_GRID_S3_ENDPOINT, FLIGHT_GRID_S3_ACCESS_KEY_ID, FLIGHT_GRID_S3_SECRET_ACCESS_KEY,
    FLIGHT_GRID_S3_REGION (default eu-west-2), FLIGHT_GRID_S3_BUCKET (default flight-grid).

    python store.py check                      # fail early if the secrets are not set
    python store.py exists KEY                 # exit 0 if present, 3 if not
    python store.py put LOCAL KEY
    python store.py get KEY LOCAL
    python store.py put-dir DIR PREFIX         # every file directly in DIR
    python store.py get-days DIR DATE [DATE..] # extract/<YYYY>/<date>.parquet into DIR/<YYYY>/ plus manifests
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REQUIRED_ENV = ("FLIGHT_GRID_S3_ENDPOINT", "FLIGHT_GRID_S3_ACCESS_KEY_ID", "FLIGHT_GRID_S3_SECRET_ACCESS_KEY")
DEFAULT_BUCKET = "flight-grid"
DEFAULT_REGION = "eu-west-2"
EXIT_MISSING = 3


def missing_env(env: dict[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    return [k for k in REQUIRED_ENV if not env.get(k)]


def bucket_name() -> str:
    return os.environ.get("FLIGHT_GRID_S3_BUCKET", DEFAULT_BUCKET)


def client() -> Any:
    import boto3  # lazy: the pure-Python parts of the repo do not need it

    missing = missing_env()
    if missing:
        raise SystemExit(f"missing environment: {', '.join(missing)}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["FLIGHT_GRID_S3_ENDPOINT"],
        aws_access_key_id=os.environ["FLIGHT_GRID_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["FLIGHT_GRID_S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("FLIGHT_GRID_S3_REGION", DEFAULT_REGION),
    )


def exists(s3: Any, key: str, bucket: str | None = None) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket or bucket_name(), Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def put(s3: Any, local: Path, key: str, bucket: str | None = None) -> None:
    s3.upload_file(str(local), bucket or bucket_name(), key)


def get(s3: Any, key: str, local: Path, bucket: str | None = None) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket or bucket_name(), key, str(local))


def put_dir(s3: Any, directory: Path, prefix: str, bucket: str | None = None) -> list[str]:
    prefix = prefix.rstrip("/") + "/"
    keys = []
    for p in sorted(Path(directory).iterdir()):
        if p.is_file():
            put(s3, p, prefix + p.name, bucket)
            keys.append(prefix + p.name)
    return keys


def day_keys(day: str) -> tuple[str, str]:
    """Object keys for a day's extract and its manifest entry."""
    return f"extract/{day[:4]}/{day}.parquet", f"extract/manifests/{day}.json"


def get_days(s3: Any, dest: Path, days: list[str], bucket: str | None = None) -> tuple[list[str], list[str]]:
    """Download extracts into dest/<YYYY>/ and manifests into dest/manifests/. Returns (got, missing)."""
    got, missing = [], []
    for d in days:
        pq_key, mf_key = day_keys(d)
        if not exists(s3, pq_key, bucket):
            missing.append(d)
            continue
        get(s3, pq_key, dest / d[:4] / f"{d}.parquet", bucket)
        get(s3, mf_key, dest / "manifests" / f"{d}.json", bucket)
        got.append(d)
    return got, missing


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        return 2
    cmd, rest = args[0], args[1:]
    if cmd == "check":
        missing = missing_env()
        if missing:
            print(f"missing secrets: {', '.join(missing)}. Add them under Settings > Secrets and variables > Actions.", file=sys.stderr)
            return 1
        client().head_bucket(Bucket=bucket_name())
        print(f"bucket {bucket_name()} reachable")
        return 0
    s3 = client()
    if cmd == "exists":
        return 0 if exists(s3, rest[0]) else EXIT_MISSING
    if cmd == "put":
        put(s3, Path(rest[0]), rest[1])
    elif cmd == "get":
        get(s3, rest[0], Path(rest[1]))
    elif cmd == "put-dir":
        print("\n".join(put_dir(s3, Path(rest[0]), rest[1])))
    elif cmd == "get-days":
        got, missing = get_days(s3, Path(rest[0]), rest[1:])
        print(f"downloaded {len(got)} days, {len(missing)} not in the bucket")
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
