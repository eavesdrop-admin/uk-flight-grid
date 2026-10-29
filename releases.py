"""Find the adsb.lol history release for a day.

Layout, verified live on 2026-10-29:

* One GitHub repo per year: ``adsblol/globe_history_<year>``.
* The repos overlap across the year boundary. The 2026 repo holds days from
  2025-12-14 onward and 2025-12-31 exists only there; the 2025 repo also holds
  late December 2024 and stops at 2025-12-30. So a day's release can sit in the
  repo for its own year or either neighbour. We search ``[Y, Y+1, Y-1]``.
* Tag: ``v<YYYY>.<MM>.<DD>-planes-readsb-<kind>-0`` where kind is ``prod``
  (canonical), ``staging`` (fallback) or ``mlatonly`` (Mode-S/MLAT-only aircraft;
  excluded from v1). There are also transient ``...-0tmp`` tags, which we ignore.
* Assets: ``<tag>.tar.aa``, ``.tar.ab`` ... (2 GB pieces of one plain tar) for
  prod and staging; a single ``<tag>.tar`` for mlatonly.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

API = "https://api.github.com"
REPO_TEMPLATE = "adsblol/globe_history_{year}"
USER_AGENT = "uk-flight-grid"

PROD = "prod"
STAGING = "staging"
MLATONLY = "mlatonly"

# Use staging when prod is more than this much smaller than staging.
PROD_SHORTFALL = 0.15

_TAG_RE = re.compile(r"^v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-(prod|staging|mlatonly)-0$")
_PART_RE = re.compile(r"\.tar(?:\.([a-z]{2}))?$")

# fetch(url) -> parsed JSON, or None for HTTP 404
Fetch = Callable[[str], object | None]


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    url: str


@dataclass(frozen=True)
class Release:
    day: date
    kind: str
    repo: str
    tag: str
    assets: tuple[Asset, ...]  # sorted by name: .tar.aa, .tar.ab, ...

    @property
    def total_bytes(self) -> int:
        return sum(a.size for a in self.assets)

    @property
    def complete(self) -> bool:
        """True when the parts form an unbroken aa, ab, ac... run (or a single .tar)."""
        suffixes = []
        for a in self.assets:
            m = _PART_RE.search(a.name)
            if not m:
                return False
            suffixes.append(m.group(1))
        if suffixes == [None]:
            return True
        want = [_split_suffix(i) for i in range(len(suffixes))]
        return suffixes == want

    def parts(self) -> list[tuple[str, int]]:
        return [(a.url, a.size) for a in self.assets]


def _split_suffix(i: int) -> str:
    return chr(ord("a") + i // 26) + chr(ord("a") + i % 26)


def tag_for(day: date, kind: str) -> str:
    return f"v{day:%Y.%m.%d}-planes-readsb-{kind}-0"


def parse_tag(tag: str) -> tuple[date, str] | None:
    """``(day, kind)`` for a canonical tag, None for anything else (e.g. ``-0tmp``)."""
    m = _TAG_RE.match(tag)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(4)
    except ValueError:
        return None


def http_json(url: str) -> object | None:
    """GET JSON from the GitHub API. Uses ``GITHUB_TOKEN`` if set (rate limit 5000/h vs 60/h)."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def candidate_repos(day: date) -> list[str]:
    """Repos that may hold ``day``, most likely first."""
    years = [day.year, day.year + 1, day.year - 1]
    return [REPO_TEMPLATE.format(year=y) for y in years]


def _release_from_json(repo: str, day: date, kind: str, data: dict) -> Release:
    assets = tuple(
        sorted(
            (Asset(a["name"], int(a["size"]), a["browser_download_url"]) for a in data.get("assets", [])),
            key=lambda a: a.name,
        )
    )
    return Release(day=day, kind=kind, repo=repo, tag=data["tag_name"], assets=assets)


def fetch_release(repo: str, day: date, kind: str, fetch: Fetch = http_json) -> Release | None:
    """The ``kind`` release for ``day`` in one repo, or None if absent/draft/incomplete."""
    tag = tag_for(day, kind)
    data = fetch(f"{API}/repos/{repo}/releases/tags/{tag}")
    if not isinstance(data, dict) or data.get("draft"):
        return None
    release = _release_from_json(repo, day, kind, data)
    if not release.assets or not release.complete:
        return None
    return release


def find_release(
    day: date, kind: str, fetch: Fetch = http_json, repos: Iterable[str] | None = None
) -> Release | None:
    """First complete ``kind`` release for ``day`` across the candidate repos."""
    for repo in repos if repos is not None else candidate_repos(day):
        release = fetch_release(repo, day, kind, fetch)
        if release is not None:
            return release
    return None


def choose_release(prod: Release | None, staging: Release | None) -> Release | None:
    """Prod unless it is missing or more than 15% smaller than staging."""
    if prod is None:
        return staging
    if staging is not None and prod.total_bytes < staging.total_bytes * (1 - PROD_SHORTFALL):
        return staging
    return prod


def resolve_day(day: date, fetch: Fetch = http_json) -> Release | None:
    """The release to extract for ``day`` (never mlatonly), or None if the day is unavailable."""
    prod = find_release(day, PROD, fetch)
    staging = find_release(day, STAGING, fetch)
    return choose_release(prod, staging)


def list_days(repo: str, fetch: Fetch = http_json) -> dict[date, dict[str, Release]]:
    """Every day in ``repo`` by tag listing (100 per page), for discovery and backfill planning."""
    out: dict[date, dict[str, Release]] = {}
    page = 1
    while True:
        data = fetch(f"{API}/repos/{repo}/releases?per_page=100&page={page}")
        if not isinstance(data, list) or not data:
            break
        for item in data:
            parsed = parse_tag(item.get("tag_name", ""))
            if parsed is None or item.get("draft"):
                continue
            day, kind = parsed
            release = _release_from_json(repo, day, kind, item)
            if release.assets and release.complete:
                out.setdefault(day, {})[kind] = release
        page += 1
    return out


def discover_days(years: Iterable[int], fetch: Fetch = http_json) -> dict[date, Release]:
    """Chosen release per day across the yearly repos, merging the overlap at year boundaries.

    A day present in several repos takes the repo whose year matches the day (the canonical
    home); otherwise the first repo that has it. Prod/staging choice then applies per day.
    """
    merged: dict[date, dict[str, Release]] = {}
    for year in sorted(set(years)):
        repo = REPO_TEMPLATE.format(year=year)
        for day, kinds in list_days(repo, fetch).items():
            slot = merged.setdefault(day, {})
            for kind, release in kinds.items():
                existing = slot.get(kind)
                if existing is None or (day.year == year and existing.repo != repo):
                    slot[kind] = release
    chosen: dict[date, Release] = {}
    for day, kinds in merged.items():
        release = choose_release(kinds.get(PROD), kinds.get(STAGING))
        if release is not None:
            chosen[day] = release
    return dict(sorted(chosen.items()))
