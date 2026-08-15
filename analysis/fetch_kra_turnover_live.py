"""Collect clean-sample race-market turnover from KRA historical result pages.

This is a recovery/reproducibility path for the compact turnover extract used by
Cultural Industry Table 4.  The author Dropbox contains archived raw JSON-gzip
pages at ``/kra-analysis/data/raw_collected_v3_15w``.  The connected Dropbox
interface confirms the archived pages contain the same historical turnover
figures as KRA's current historical result endpoints.  We therefore collect only
four race-market totals needed by Table 4 and freeze the compact extract in Git.

No odds or participant-level data are downloaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://race.kra.co.kr/raceScore/"
ENDPOINTS = {
    "quinella": "ScoretableBettingprofitScm.do",
    "exacta": "ScoretableBettingprofitBoth.do",
    "trio": "ScoretableBettingprofit3Bc.do",
    "trifecta": "ScoretableBettingprofit3Both.do",
}
LABELS = {
    "quinella": "복승식",
    "exacta": "쌍승식",
    "trio": "삼복승식",
    "trifecta": "삼쌍승식",
}

# Independent provenance check against the archived Dropbox raw file
# /kra-analysis/data/raw_collected_v3_15w/kra_2025/raw_archive/2025/2025-12/
# 2025-12-13_2_03.json.gz, inspected through the connected Dropbox account.
ARCHIVE_CHECK_RACE = "2025-12-13_2_03"
ARCHIVE_CHECK = {
    "quinella": 763_893_500,
    "exacta": 195_364_900,
    "trio": 839_141_700,
    "trifecta": 411_991_600,
}

_thread_state = threading.local()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--race-ids",
        type=Path,
        default=Path("outputs/final_19301/clean_race_ids.txt"),
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/turnover_by_race_market.csv.gz")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/turnover_by_race_market.manifest.json"),
    )
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--polite-delay",
        type=float,
        default=0.08,
        help="minimum per-worker delay before each request",
    )
    return p.parse_args()


def race_parts(race_id: str) -> tuple[str, str, int]:
    date_text, meet, race_no = race_id.split("_")
    return date_text.replace("-", ""), meet, int(race_no)


def url_for(race_id: str, market: str) -> str:
    date_text, meet, race_no = race_parts(race_id)
    endpoint = ENDPOINTS[market]
    return (
        f"{BASE_URL}{endpoint}?meet={meet}&realRcDate={date_text}&realRcNo={race_no}"
    )


def decode_kra(raw: bytes) -> str:
    for enc in ("euc-kr", "cp949", "utf-8"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if "경주" in text or "매출" in text:
            return text
    return raw.decode("cp949", errors="replace")


def page_text(raw: bytes) -> str:
    text = decode_kra(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).replace("\xa0", " ").split())


def parse_turnover(raw: bytes, market: str) -> int:
    text = page_text(raw)
    label = LABELS[market]
    if market == "quinella":
        # The Scm page reports win/place/quinella totals in one footer.
        pattern = rf"(?<![가-힣]){re.escape(label)}\s*:\s*([0-9,]+)\s*원"
    else:
        pattern = (
            rf"(?<![가-힣]){re.escape(label)}\s*매출총액\s*:?\s*"
            rf"([0-9,]+)\s*원"
        )
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"{market}: turnover label not found")
    return int(match.group(1).replace(",", ""))


def fetch_bytes(url: str, *, retries: int, timeout: float, polite_delay: float) -> bytes:
    # Each worker maintains its own most-recent request time, avoiding a bursty
    # tight loop while still allowing modest parallelism.
    last = getattr(_thread_state, "last_request", 0.0)
    wait = polite_delay - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    headers = {
        "User-Agent": "Mozilla/5.0 (KRA academic reproducibility audit; historical pages)",
        "Accept": "text/html,application/xhtml+xml",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
            _thread_state.last_request = time.monotonic()
            if len(raw) < 10_000:
                raise IOError(f"unexpectedly short KRA response: {len(raw)} bytes")
            return raw
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            # Deterministic base backoff plus a tiny jitter to avoid synchronized
            # retries across workers.
            time.sleep(min(20.0, 0.8 * (2**attempt)) + random.random() * 0.2)
    raise RuntimeError(f"KRA request failed after {retries} attempts: {url}: {last_error}")


def collect_race(
    race_id: str, *, retries: int, timeout: float, polite_delay: float
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for market in ("quinella", "exacta", "trio", "trifecta"):
        url = url_for(race_id, market)
        raw = fetch_bytes(
            url, retries=retries, timeout=timeout, polite_delay=polite_delay
        )
        turnover = parse_turnover(raw, market)
        if turnover <= 0 or turnover % 100 != 0:
            raise ValueError(f"{race_id}/{market}: invalid turnover {turnover}")
        rows.append(
            {
                "race_id": race_id,
                "market": market,
                "turnover_won": turnover,
            }
        )
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_output(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (str(r["race_id"]), str(r["market"])))
    with gzip.open(path, "wt", encoding="utf-8", newline="", mtime=0) as f:
        writer = csv.DictWriter(f, fieldnames=["race_id", "market", "turnover_won"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    race_ids = [
        line.strip()
        for line in args.race_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(race_ids) != 3338 or len(set(race_ids)) != 3338:
        raise AssertionError(f"expected 3,338 unique clean race IDs, got {len(set(race_ids))}")

    # Fail fast on one independently archived race before launching the full job.
    archived_rows = collect_race(
        ARCHIVE_CHECK_RACE,
        retries=args.retries,
        timeout=args.timeout,
        polite_delay=args.polite_delay,
    )
    archived = {str(r["market"]): int(r["turnover_won"]) for r in archived_rows}
    if archived != ARCHIVE_CHECK:
        raise AssertionError(f"live/archive turnover check failed: {archived} != {ARCHIVE_CHECK}")
    print(f"PASS: KRA live historical page matches Dropbox archive for {ARCHIVE_CHECK_RACE}")

    all_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                collect_race,
                race_id,
                retries=args.retries,
                timeout=args.timeout,
                polite_delay=args.polite_delay,
            ): race_id
            for race_id in race_ids
        }
        completed = 0
        for future in as_completed(futures):
            race_id = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # retain all failure evidence before aborting
                failures.append((race_id, repr(exc)))
            completed += 1
            if completed % 100 == 0 or completed == len(race_ids):
                elapsed = time.monotonic() - start
                print(
                    f"completed={completed:,}/{len(race_ids):,} "
                    f"failures={len(failures)} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

    if failures:
        for race_id, err in failures[:30]:
            print("FAIL", race_id, err)
        raise RuntimeError(f"turnover collection failed for {len(failures)} races")
    if len(all_rows) != 3338 * 4:
        raise AssertionError(f"expected {3338*4:,} turnover rows, got {len(all_rows):,}")

    write_output(all_rows, args.output)
    output_hash = sha256(args.output)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "clean_races": 3338,
            "markets": ["quinella", "exacta", "trio", "trifecta"],
            "rows": len(all_rows),
        },
        "source": {
            "primary": "KRA historical raceScore pages",
            "base_url": BASE_URL,
            "endpoints": ENDPOINTS,
            "dropbox_archive_root": "/kra-analysis/data/raw_collected_v3_15w",
            "dropbox_archive_crosscheck_race": ARCHIVE_CHECK_RACE,
            "dropbox_archive_crosscheck_values_won": ARCHIVE_CHECK,
            "dropbox_archive_crosscheck_passed": True,
        },
        "collection": {
            "workers": args.workers,
            "retries": args.retries,
            "timeout_seconds": args.timeout,
            "polite_delay_seconds_per_worker": args.polite_delay,
        },
        "output": {
            "path": args.output.as_posix(),
            "sha256": output_hash,
            "compression": "gzip",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote {len(all_rows):,} turnover rows -> {args.output} sha256={output_hash}")


if __name__ == "__main__":
    main()
