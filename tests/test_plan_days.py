import json
from datetime import date

import plan_days as pd


def test_window_is_twelve_complete_months_before_this_one():
    assert pd.window(date(2026, 10, 29)) == (date(2025, 10, 1), date(2026, 9, 30))
    assert pd.window(date(2026, 1, 1)) == (date(2025, 1, 1), date(2025, 12, 31))
    assert pd.window(date(2026, 3, 15)) == (date(2025, 3, 1), date(2026, 2, 28))


def test_sampled_days_follow_the_ordinal_and_are_about_a_quarter():
    days = pd.sampled(date(2025, 10, 1), date(2026, 9, 30), 4)
    assert len(days) == 91
    assert all(d.toordinal() % 4 == 0 for d in days)
    assert all((b - a).days == 4 for a, b in zip(days, days[1:]))


def test_sample_is_the_same_whatever_the_window():
    a = set(pd.sampled(date(2025, 10, 1), date(2026, 9, 30), 4))
    b = set(pd.sampled(date(2026, 9, 1), date(2026, 9, 30), 4))
    assert b <= a


def test_the_pilot_month_contains_its_share_of_the_sample():
    month = pd.month_days(2026, 9)
    assert len(month) == 30
    in_sample = pd.sampled_in([d.isoformat() for d in month], 4)
    assert set(in_sample) == {d.isoformat() for d in pd.sampled(date(2026, 9, 1), date(2026, 9, 30), 4)}
    assert pd.sampled_in(["2026-09-01"], 1) == ["2026-09-01"]


def test_stride_one_is_every_day():
    assert len(pd.sampled(date(2026, 9, 1), date(2026, 9, 30), 1)) == 30


def test_cli_outputs_json_and_enforces_the_matrix_limit(capsys):
    assert pd.main(["sampled", "--today", "2026-10-29"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 182  # default stride 2
    assert pd.main(["sampled", "--today", "2026-10-29", "--stride", "1"]) == 1  # 365 days > 256
    assert pd.main(["pilot", "--month", "2026-02"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 28
    assert pd.main(["dates", "2026-09-05, 2026-09-01"]) == 0
    assert json.loads(capsys.readouterr().out) == ["2026-09-01", "2026-09-05"]
