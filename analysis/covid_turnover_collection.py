from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


BASE_URL = "https://race.kra.co.kr/raceScore/ScoretableDetailList.do"
MEETS = {1: "seoul", 2: "jeju", 3: "busan"}
MARKET_LABELS = {
    "단승식": "win",
    "연승식": "place",
    "복승식": "quinella",
    "쌍승식": "exacta",
    "복연승식": "quinella_place",
    "삼복승식": "trio",
    "삼쌍승식": "trifecta",
    "합계": "total",
}
NUMBER_RE = re.compile(r"([0-9][0-9,]*)")


@dataclass(frozen=True)
class FetchResult:
    race_date: str
    meet: int
    meet_name: str
    values: dict[str, int]
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect KRA first-race turnover by meeting/date. "
            "Used to quantify the abnormal 2020-2021 COVID-period market scale."
        )
    )
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/covid_turnover_race1.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/covid_turnover_race1_summary.csv"),
    )
    return parser.parse_args()


def parse_iso(value: str) -> date:
    return date.fromisoformat(value)


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_url(meet: int, race_date: date) -> str:
    params = {
        "meet": meet,
        "realRcDate": race_date.strftime("%Y%m%d"),
        "realRcNo": 1,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def request_text(url: str, timeout: float, retries: int) -> str | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; KRA-academic-research/1.0; "
            "+https://github.com/Joocheol/kra-post-trifecta)"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                raw = response.read()
            return raw.decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            if attempt >= retries:
                return None
            time.sleep(0.25 * (attempt + 1))
    return None


def parse_turnover(html: str) -> dict[str, int] | None:
    if "매출액" not in html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    target = None
    for table in soup.find_all("table"):
        text = " ".join(table.stripped_strings)
        if "매출액" in text and "단승식" in text and "삼쌍승식" in text:
            target = text
            break
    if target is None:
        return None

    values: dict[str, int] = {}
    for korean, key in MARKET_LABELS.items():
        pattern = re.compile(re.escape(korean) + r"\s*:?\s*([0-9][0-9,]*)")
        match = pattern.search(target)
        if match:
            values[key] = int(match.group(1).replace(",", ""))
    required = {"win", "place", "quinella", "exacta", "quinella_place", "trio", "trifecta", "total"}
    if not required.issubset(values):
        return None
    return values


def fetch_one(meet: int, race_date: date, timeout: float, retries: int) -> FetchResult | None:
    url = build_url(meet, race_date)
    html = request_text(url, timeout, retries)
    if not html:
        return None
    values = parse_turnover(html)
    if values is None:
        return None

    # Guard against a site fallback returning a different date.
    ymd_korean = f"{race_date.year}년 {race_date.month:02d}월 {race_date.day:02d}일"
    if ymd_korean not in html:
        return None
    return FetchResult(
        race_date=race_date.isoformat(),
        meet=meet,
        meet_name=MEETS[meet],
        values=values,
        url=url,
    )


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def write_rows(path: Path, results: list[FetchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "race_date", "year", "meet", "meet_name",
        "win", "place", "quinella", "exacta", "quinella_place", "trio", "trifecta", "total", "url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(results, key=lambda x: (x.race_date, x.meet)):
            row = {
                "race_date": item.race_date,
                "year": item.race_date[:4],
                "meet": item.meet,
                "meet_name": item.meet_name,
                **item.values,
                "url": item.url,
            }
            writer.writerow(row)


def write_summary(path: Path, results: list[FetchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[FetchResult]] = {}
    for item in results:
        year = item.race_date[:4]
        groups.setdefault((year, "all"), []).append(item)
        groups.setdefault((year, item.meet_name), []).append(item)

    fields = [
        "year", "meet_name", "n_meeting_dates",
        "total_mean", "total_median", "total_q25", "total_q75", "total_min", "total_max",
        "win_median", "quinella_median", "exacta_median", "trio_median", "trifecta_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (year, meet_name), items in sorted(groups.items()):
            totals = [x.values["total"] for x in items]
            row = {
                "year": year,
                "meet_name": meet_name,
                "n_meeting_dates": len(items),
                "total_mean": round(sum(totals) / len(totals), 2),
                "total_median": round(median(totals), 2),
                "total_q25": round(percentile(totals, 0.25), 2),
                "total_q75": round(percentile(totals, 0.75), 2),
                "total_min": min(totals),
                "total_max": max(totals),
                "win_median": round(median([x.values["win"] for x in items]), 2),
                "quinella_median": round(median([x.values["quinella"] for x in items]), 2),
                "exacta_median": round(median([x.values["exacta"] for x in items]), 2),
                "trio_median": round(median([x.values["trio"] for x in items]), 2),
                "trifecta_median": round(median([x.values["trifecta"] for x in items]), 2),
            }
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    start = parse_iso(args.start_date)
    end = parse_iso(args.end_date)
    if end < start:
        raise SystemExit("end date must be on or after start date")

    jobs = [(meet, d) for d in iter_dates(start, end) for meet in MEETS]
    results: list[FetchResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(fetch_one, meet, d, args.timeout, args.retries): (meet, d)
            for meet, d in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    if not results:
        raise SystemExit("No KRA race-1 turnover pages were found")
    write_rows(args.output, results)
    write_summary(args.summary_output, results)
    print(f"Collected {len(results)} meeting-date first-race pages")
    print(f"Rows: {args.output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
