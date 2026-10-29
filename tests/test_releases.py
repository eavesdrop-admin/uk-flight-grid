from __future__ import annotations

from datetime import date

import releases as rl


def asset(tag: str, suffix: str, size: int, repo: str) -> dict:
    name = f"{tag}.tar{suffix}"
    return {"name": name, "size": size, "browser_download_url": f"https://example.invalid/{repo}/{tag}/{name}"}


def release_json(repo: str, day: date, kind: str, sizes=(2000, 1800), draft=False, tag=None) -> dict:
    tag = tag or rl.tag_for(day, kind)
    if kind == "mlatonly":
        assets = [asset(tag, "", sizes[0], repo)]
    else:
        assets = [asset(tag, f".{rl._split_suffix(i)}", s, repo) for i, s in enumerate(sizes)]
    return {"tag_name": tag, "draft": draft, "assets": assets}


class FakeGitHub:
    """Mocked adsb.lol repos: {repo: [release json, ...]}; answers the two endpoints we use."""

    def __init__(self, repos: dict[str, list[dict]]):
        self.repos = repos
        self.calls: list[str] = []

    def __call__(self, url: str):
        self.calls.append(url)
        path = url.removeprefix(rl.API)
        parts = path.split("/")  # '', repos, owner, name, releases, ...
        repo = f"{parts[2]}/{parts[3]}"
        if repo not in self.repos:
            return None
        if "/releases/tags/" in path:
            tag = path.split("/releases/tags/")[1]
            return next((r for r in self.repos[repo] if r["tag_name"] == tag), None)
        query = dict(q.split("=") for q in path.split("?")[1].split("&"))
        per, page = int(query["per_page"]), int(query["page"])
        return self.repos[repo][(page - 1) * per : page * per]


def repo_name(year: int) -> str:
    return rl.REPO_TEMPLATE.format(year=year)


def world() -> FakeGitHub:
    """Year boundary as observed live: late Dec appears in both repos, Dec 31 only in the next."""
    r25, r26, r24 = repo_name(2025), repo_name(2026), repo_name(2024)
    d = date
    return FakeGitHub(
        {
            r24: [release_json(r24, d(2024, 12, 30), "prod"), release_json(r24, d(2024, 12, 30), "staging")],
            r25: [
                release_json(r25, d(2024, 12, 31), "prod"),
                release_json(r25, d(2025, 6, 1), "prod"),
                release_json(r25, d(2025, 12, 30), "prod", sizes=(2000, 1000)),
                release_json(r25, d(2025, 12, 30), "mlatonly"),
            ],
            r26: [
                release_json(r26, d(2025, 12, 30), "prod", sizes=(2000, 1900)),
                release_json(r26, d(2025, 12, 31), "prod"),
                release_json(r26, d(2025, 12, 31), "staging", sizes=(2000, 1500)),
                release_json(r26, d(2026, 1, 1), "prod"),
                release_json(r26, d(2026, 1, 1), "prod", tag="v2026.01.01-planes-readsb-prod-0tmp"),
            ],
        }
    )


def test_tag_format_and_parse_round_trip():
    tag = rl.tag_for(date(2026, 10, 27), "prod")
    assert tag == "v2026.10.27-planes-readsb-prod-0"
    assert rl.parse_tag(tag) == (date(2026, 10, 27), "prod")
    assert rl.parse_tag("v2026.10.27-planes-readsb-prod-0tmp") is None
    assert rl.parse_tag("v2026.13.40-planes-readsb-prod-0") is None
    assert rl.parse_tag("nightly") is None


def test_split_suffixes():
    assert [rl._split_suffix(i) for i in (0, 1, 25, 26)] == ["aa", "ab", "az", "ba"]


def test_candidate_repos_search_own_year_then_neighbours():
    assert rl.candidate_repos(date(2025, 12, 31)) == [repo_name(2025), repo_name(2026), repo_name(2024)]


def test_year_rollover_dec_31_only_exists_in_next_years_repo():
    got = rl.resolve_day(date(2025, 12, 31), world())
    assert got is not None
    assert got.repo == repo_name(2026)
    assert got.kind == "prod"
    assert [a.name for a in got.assets] == [
        "v2025.12.31-planes-readsb-prod-0.tar.aa",
        "v2025.12.31-planes-readsb-prod-0.tar.ab",
    ]


def test_year_rollover_overlap_day_prefers_its_own_years_repo():
    fetch = world()
    # prod is 3000 in the 2025 repo (smaller), but we still take the day's own repo first
    got = rl.find_release(date(2025, 12, 30), "prod", fetch)
    assert got.repo == repo_name(2025)
    assert fetch.calls[0].endswith("/repos/adsblol/globe_history_2025/releases/tags/v2025.12.30-planes-readsb-prod-0")


def test_january_first_is_in_the_new_years_repo():
    assert rl.resolve_day(date(2026, 1, 1), world()).repo == repo_name(2026)


def test_late_december_of_previous_year_found_in_next_repo_when_own_repo_lacks_it():
    got = rl.resolve_day(date(2024, 12, 31), world())
    assert got is not None and got.repo == repo_name(2025)


def test_missing_day_returns_none():
    assert rl.resolve_day(date(2030, 1, 1), world()) is None


def test_mlatonly_is_never_chosen():
    fetch = FakeGitHub({repo_name(2025): [release_json(repo_name(2025), date(2025, 3, 3), "mlatonly")]})
    assert rl.resolve_day(date(2025, 3, 3), fetch) is None


def test_choose_prefers_prod():
    r = repo_name(2026)
    prod = rl._release_from_json(r, date(2026, 1, 5), "prod", release_json(r, date(2026, 1, 5), "prod", sizes=(2000, 1800)))
    staging = rl._release_from_json(r, date(2026, 1, 5), "staging", release_json(r, date(2026, 1, 5), "staging", sizes=(2000, 1500)))
    assert rl.choose_release(prod, staging) is prod
    assert rl.choose_release(None, staging) is staging
    assert rl.choose_release(prod, None) is prod
    assert rl.choose_release(None, None) is None


def test_choose_staging_when_prod_is_more_than_15_percent_smaller():
    r = repo_name(2026)
    day = date(2026, 1, 5)

    def mk(kind, total):
        return rl._release_from_json(r, day, kind, release_json(r, day, kind, sizes=(total,)))

    staging = mk("staging", 1000)
    assert rl.choose_release(mk("prod", 840), staging).kind == "staging"   # 16% smaller
    assert rl.choose_release(mk("prod", 860), staging).kind == "prod"      # 14% smaller
    assert rl.choose_release(mk("prod", 1500), staging).kind == "prod"     # bigger is fine


def test_resolve_day_applies_prod_vs_staging_rule():
    day = date(2025, 12, 31)
    fetch = world()
    assert rl.resolve_day(day, fetch).kind == "prod"  # 3800 vs 3500
    r26 = repo_name(2026)
    fetch.repos[r26][1] = release_json(r26, day, "prod", sizes=(2000, 900))  # 2900 < 0.85 * 3500 = 2975
    assert rl.resolve_day(day, fetch).kind == "staging"


def test_incomplete_part_runs_and_drafts_are_rejected():
    r = repo_name(2026)
    day = date(2026, 2, 2)
    broken = release_json(r, day, "prod")
    broken["assets"] = [broken["assets"][1]]  # .ab without .aa
    empty = release_json(r, day, "staging")
    empty["assets"] = []
    draft = release_json(r, date(2026, 2, 3), "prod", draft=True)
    fetch = FakeGitHub({r: [broken, empty, draft]})
    assert rl.resolve_day(day, fetch) is None
    assert rl.resolve_day(date(2026, 2, 3), fetch) is None


def test_single_tar_asset_is_complete():
    r = repo_name(2026)
    rel = rl._release_from_json(r, date(2026, 1, 1), "mlatonly", release_json(r, date(2026, 1, 1), "mlatonly"))
    assert rel.complete


def test_parts_are_url_size_pairs_in_order():
    got = rl.resolve_day(date(2026, 1, 1), world())
    assert [s for _, s in got.parts()] == [2000, 1800]
    assert got.total_bytes == 3800
    assert got.parts()[0][0].endswith(".tar.aa")


def test_list_days_paginates_and_ignores_tmp_tags():
    r = repo_name(2026)
    days = rl.list_days(r, world())
    assert date(2026, 1, 1) in days
    assert set(days[date(2026, 1, 1)]) == {"prod"}  # the -0tmp tag is ignored
    big = FakeGitHub({r: [release_json(r, date(2026, 1, 1 + i % 28), "prod", sizes=(1,)) for i in range(250)]})
    rl.list_days(r, big)
    assert sum("page=" in c for c in big.calls) == 4  # 100 + 100 + 50 + empty page


def test_discover_days_merges_overlap_across_both_repos():
    fetch = world()
    days = rl.discover_days([2025, 2026], fetch)
    assert days[date(2025, 12, 30)].repo == repo_name(2025)  # own year wins the overlap
    assert days[date(2025, 12, 31)].repo == repo_name(2026)  # only in the 2026 repo
    assert days[date(2026, 1, 1)].repo == repo_name(2026)
    assert days[date(2024, 12, 31)].repo == repo_name(2025)
    assert list(days) == sorted(days)
