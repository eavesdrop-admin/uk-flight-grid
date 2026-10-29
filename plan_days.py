"""Which days to extract and build. Pure date arithmetic, no network.

    python plan_days.py sampled [--stride 4] [--end 2026-09-30]   # the production sample
    python plan_days.py pilot --month 2026-09                     # every day of one month
    python plan_days.py dates 2026-09-01,2026-09-05               # an explicit list
    python plan_days.py window                                    # start and end of the window

The window is the 12 complete calendar months ending at the end of the previous month.
Sampled days are those whose proleptic-Gregorian ordinal is divisible by the stride, so the
same calendar days are picked whatever window they fall in: the pilot month's sampled days
are a subset of the production sample and are extracted once. Output is a JSON array
(``--lines`` for one date per line).
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import date, timedelta

WINDOW_MONTHS = 12
DEFAULT_STRIDE = 4
MAX_MATRIX_JOBS = 256  # GitHub Actions limit on a matrix


def window(today: date, months: int = WINDOW_MONTHS) -> tuple[date, date]:
    """First and last day of the `months` complete calendar months before `today`'s month."""
    end = today.replace(day=1) - timedelta(days=1)
    y, m = end.year, end.month - (months - 1)
    while m < 1:
        m += 12
        y -= 1
    return date(y, m, 1), end


def month_days(year: int, month: int) -> list[date]:
    return [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]


def sampled(start: date, end: date, stride: int = DEFAULT_STRIDE) -> list[date]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    n = (end - start).days + 1
    return [d for d in (start + timedelta(days=i) for i in range(n)) if d.toordinal() % stride == 0]


def sampled_in(days: list[str], stride: int) -> list[str]:
    """The ISO dates in `days` that belong to the stride sample."""
    return [d for d in days if date.fromisoformat(d).toordinal() % stride == 0]


def _emit(days: list[date], lines: bool) -> None:
    out = [d.isoformat() for d in sorted(set(days))]
    print("\n".join(out) if lines else json.dumps(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sampled")
    s.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    s.add_argument("--end", type=date.fromisoformat, help="last day of the window (default: end of last month)")
    s.add_argument("--months", type=int, default=WINDOW_MONTHS)
    p = sub.add_parser("pilot")
    p.add_argument("--month", required=True, help="YYYY-MM")
    d = sub.add_parser("dates")
    d.add_argument("list", help="comma-separated YYYY-MM-DD")
    w = sub.add_parser("window")
    w.add_argument("--months", type=int, default=WINDOW_MONTHS)
    for sp in (s, p, d, w):
        sp.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
        sp.add_argument("--lines", action="store_true")
    a = ap.parse_args(argv)

    today = a.today or date.today()
    if a.cmd == "window":
        start, end = window(today, a.months)
        print(f"{start.isoformat()}\n{end.isoformat()}" if a.lines else json.dumps({"start": start.isoformat(), "end": end.isoformat()}))
        return 0
    if a.cmd == "sampled":
        start, end = window(a.end + timedelta(days=1) if a.end else today, a.months)
        days = sampled(start, end, a.stride)
    elif a.cmd == "pilot":
        y, m = (int(x) for x in a.month.split("-"))
        days = month_days(y, m)
    else:
        days = [date.fromisoformat(x.strip()) for x in a.list.split(",") if x.strip()]
    if len(set(days)) > MAX_MATRIX_JOBS:
        print(f"{len(set(days))} days exceeds the {MAX_MATRIX_JOBS}-job matrix limit; extract in chunks", file=sys.stderr)
        return 1
    _emit(days, a.lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
