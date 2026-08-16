#!/usr/bin/env python3
"""Build a compact race-by-market turnover file from archived KRA JSON gzip files.

This deliberately avoids reparsing the large odds tables. It only reads the archived
HTML pages, extracts posted race-market total turnover with ``analysis.turnover``,
and freezes a small derived data set that can be committed for later re-verification.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis.turnover import extract_turnovers

START_DATE = "2016-06-10"
END_DATE = "2025-12-31"
UNAVAILABLE_YEARS = {2020, 2021}
MARKETS = ("win", "exacta", "quinella", "trio", "trifecta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/turnover_by_race_market.csv.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/turnover_by_race_market.manifest.json"),
    )
    return parser.parse_args()


def race_id_from_path(path: Path) -> str:
    if path.name.endswith(".json.gz"):
        return path.name[:-8]
    return path.stem


def in_scope(race_id: str) -> bool:
    try:
        race_date = race_id.split("_")[0]
        year = int(race_date[:4])
    except Exception:
        return False
    return START_DATE <= race_date <= END_DATE and year not in UNAVAILABLE_YEARS


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_files(raw_root: Path) -> list[Path]:
    files = sorted(raw_root.glob("kra_*/raw_archive/*/*/*.json.gz"))
    files = [path for path in files if in_scope(race_id_from_path(path))]
    if not files:
        raise FileNotFoundError(
            f"no in-scope *.json.gz files under {raw_root}; expected "
            "kra_*/raw_archive/YYYY/YYYY-MM/*.json.gz"
        )
    return files


def main() -> None:
    args = parse_args()
    files = locate_files(args.raw_root)
    rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    raw_digest = hashlib.sha256()

    for index, path in enumerate(files, 1):
        race_id = race_id_from_path(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        turnovers = extract_turnovers(payload.get("pages", {}))
        race_date = race_id.split("_")[0]
        for market in MARKETS:
            value = turnovers.get(market)
            if value is None:
                missing.append({"race_id": race_id, "market": market})
                continue
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "market": market,
                    "turnover_won": int(value),
                }
            )
        raw_digest.update(path.relative_to(args.raw_root).as_posix().encode("utf-8"))
        raw_digest.update(b"\0")
        raw_digest.update(file_sha256(path).encode("ascii"))
        raw_digest.update(b"\n")
        if index % 1000 == 0:
            print(f"read {index:,}/{len(files):,} raw races")

    frame = pd.DataFrame(rows).sort_values(["race_id", "market"]).reset_index(drop=True)
    if frame.duplicated(["race_id", "market"]).any():
        raise AssertionError("duplicate race-market turnover rows")
    if (frame["turnover_won"] <= 0).any():
        raise AssertionError("non-positive turnover found")

    race_count = frame["race_id"].nunique()
    if race_count != len(files):
        missing_races = len(files) - race_count
        raise AssertionError(f"turnover file misses all markets for {missing_races} races")
    if missing:
        miss = pd.DataFrame(missing)
        print(miss.groupby("market").size().to_string())
        raise AssertionError(
            f"{len(missing):,} race-market turnover values are missing; "
            "do not freeze an incomplete data set"
        )

    expected_rows = len(files) * len(MARKETS)
    if len(frame) != expected_rows:
        raise AssertionError(f"expected {expected_rows:,} rows, got {len(frame):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, compression="gzip")
    output_sha = file_sha256(args.output)
    manifest = {
        "source": "Korea Racing Authority archived race-result HTML stored as JSON gzip",
        "raw_root": str(args.raw_root),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "unavailable_years": sorted(UNAVAILABLE_YEARS),
        "markets": list(MARKETS),
        "n_raw_races": len(files),
        "n_rows": len(frame),
        "raw_file_inventory_sha256": raw_digest.hexdigest(),
        "output": args.output.as_posix(),
        "output_sha256": output_sha,
        "definition": "posted race-market total turnover in won; one row per race and market",
        "parser": "analysis.turnover.extract_turnovers",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: {len(files):,} races, {len(frame):,} turnover rows -> {args.output}")


if __name__ == "__main__":
    main()
