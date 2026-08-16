"""Collect final clean-sample race-market turnover for Cultural Industry Table 4.

The author's connected Dropbox keeps the archived raw KRA JSON-gzip pages under
``/kra-analysis/data/raw_collected_v3_15w``.  We independently inspected an
archived race and confirmed its race-market turnovers against KRA's historical
``ScoretableDetailList`` page.  That detail page reports all market totals in one
request, so this recovery path uses one historical page per clean race, freezes
only the four totals needed by Table 4, and records the Dropbox cross-check in the
manifest.  No odds or participant-level data are downloaded here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
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

BASE_URL = "https://race.kra.co.kr/raceScore/ScoretableDetailList.do"
MARKETS = ("quinella", "exacta", "trio", "trifecta")
LABELS = {
    "quinella": "복승식",
    "exacta": "쌍승식",
    "trio": "삼복승식",
    "trifecta": "삼쌍승식",
}

# Independently inspected in connected Dropbox:
# /kra-analysis/data/raw_collected_v3_15w/kra_2025/raw_archive/2025/2025-12/
# 2025-12-13_2_03.json.gz
ARCHIVE_CHECK_RACE = "2025-12-13_2_03"
ARCHIVE_CHECK = {
    "quinella": 763_893_500,
    "exacta": 195_364_900,
    "trio": 839_141_700,
    "trifecta": 411_991_600,
}

_thread_state = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--race-ids",
        type=Path,
        default=Path("outputs/final_19301/clean_race_ids.txt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/turnover_by_race_market.csv.gz")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/turnover_by_race_market.manifest.json"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--polite-delay", type=float, default=0.10)
    return parser.parse_args()


def race_parts(race_id: str) -> tuple[str, str, int]:
    date_text, meet, race_no = race_id.split("_")
    return date_text.replace("-", ""), meet, int(race_no)


def url_for(race_id: str) -> str:
    date_text, meet, race_no = race_parts(race_id)
    return f"{BASE_URL}?meet={meet}&realRcDate={date_text}&realRcNo={race_no}"


def decode_kra(raw: bytes) -> str:
    for encoding in ("euc-kr", "cp949", "utf-8"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "경주" in text or "매출" in text:
            return text
    return raw.decode("cp949", errors="replace")


def page_text(raw: bytes) -> str:
    text = re.sub(r"<[^>]+>", " ", decode_kra(raw))
    return " ".join(html.unescape(text).replace("\xa0", " ").split())


def parse_turnovers(raw: bytes) -> dict[str, int]:
    text = page_text(raw)
    result: dict[str, int] = {}
    for market, label in LABELS.items():
        match = re.search(
            rf"(?<![가-힣]){re.escape(label)}\s*:\s*([0-9,]+)(?:\s*원)?",
            text,
        )
        if not match:
            raise ValueError(f"{market}: turnover label not found on detail page")
        value = int(match.group(1).replace(",", ""))
        if value <= 0 or value % 100 != 0:
            raise ValueError(f"{market}: invalid turnover {value}")
        result[market] = value
    return result


def fetch_bytes(url: str, *, retries: int, timeout: float, polite_delay: float) -> bytes:
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
            time.sleep(min(20.0, 0.8 * (2**attempt)) + random.random() * 0.2)
    raise RuntimeError(f"KRA request failed after {retries} attempts: {url}: {last_error}")


def collect_race(
    race_id: str, *, retries: int, timeout: float, polite_delay: float
) -> list[dict[str, object]]:
    raw = fetch_bytes(
        url_for(race_id),
        retries=retries,
        timeout=timeout,
        polite_delay=polite_delay,
    )
    values = parse_turnovers(raw)
    return [
        {"race_id": race_id, "market": market, "turnover_won": values[market]}
        for market in MARKETS
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_output(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (str(row["race_id"]), str(row["market"])))
    with path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_file:
                writer = csv.DictWriter(
                    text_file, fieldnames=["race_id", "market", "turnover_won"]
                )
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

    archived_rows = collect_race(
        ARCHIVE_CHECK_RACE,
        retries=args.retries,
        timeout=args.timeout,
        polite_delay=args.polite_delay,
    )
    archived = {str(row["market"]): int(row["turnover_won"]) for row in archived_rows}
    if archived != ARCHIVE_CHECK:
        raise AssertionError(f"KRA/Dropbox archive parity failed: {archived} != {ARCHIVE_CHECK}")
    print(f"PASS: KRA detail page matches Dropbox archive for {ARCHIVE_CHECK_RACE}")

    all_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    started = time.monotonic()
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
        for completed, future in enumerate(as_completed(futures), start=1):
            race_id = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                failures.append((race_id, repr(exc)))
            if completed % 100 == 0 or completed == len(race_ids):
                elapsed = (time.monotonic() - started) / 60.0
                print(
                    f"completed={completed:,}/{len(race_ids):,} failures={len(failures)} "
                    f"elapsed={elapsed:.1f}m",
                    flush=True,
                )

    if failures:
        for race_id, error in failures[:40]:
            print("FAIL", race_id, error)
        raise RuntimeError(f"turnover collection failed for {len(failures)} races")
    expected_rows = 3338 * len(MARKETS)
    if len(all_rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows:,} rows, got {len(all_rows):,}")

    write_output(all_rows, args.output)
    output_hash = sha256(args.output)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "clean_races": 3338,
            "markets": list(MARKETS),
            "rows": len(all_rows),
        },
        "source": {
            "primary": "KRA ScoretableDetailList historical page",
            "base_url": BASE_URL,
            "one_request_per_race": True,
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
            "compression": "gzip with deterministic mtime=0",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: wrote {len(all_rows):,} turnover rows -> {args.output} sha256={output_hash}")


if __name__ == "__main__":
    main()
