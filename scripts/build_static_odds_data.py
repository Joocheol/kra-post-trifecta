#!/usr/bin/env python3
"""Precompute date-level odds files for static hosting."""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PARSED_ROOT = ROOT / "KRA" / "parsed"
OUT_ROOT = ROOT / "docs" / "data" / "odds"
MARKETS = ["win", "place", "quinella", "exacta", "quinella_place", "trio", "trifecta"]
MARKET_ORDER = {market: index for index, market in enumerate(MARKETS)}


def market_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("market="):
            return part.split("=", 1)[1]
    return None


def combo_for_row(row: dict) -> list[int]:
    combo = []
    for key in ("horse_no", "first_no", "second_no", "third_no"):
        value = row.get(key)
        if value is not None:
            combo.append(value)
    return combo


def compact_row(row: dict, market: str) -> list:
    return [
        row.get("race_id"),
        row.get("meet"),
        market,
        combo_for_row(row),
        row.get("odds"),
        row.get("is_hit"),
        row.get("is_capped_odds"),
        row.get("arrival_order"),
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for stale_path in OUT_ROOT.glob("*.json.gz"):
        stale_path.unlink()
    manifest_path = OUT_ROOT / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    rows_by_date: dict[str, list] = defaultdict(list)
    files = sorted(PARSED_ROOT.glob("kra_*/market=*/year=*/month=*/*.parquet"))

    for index, path in enumerate(files, start=1):
        market = market_from_path(path)
        if market not in MARKETS:
            continue
        table = pq.read_table(path)
        if "race_date" not in table.column_names:
            continue
        for race_date in sorted(set(table["race_date"].to_pylist())):
            filtered = table.filter(pc.equal(table["race_date"], race_date))
            rows_by_date[race_date].extend(compact_row(row, market) for row in filtered.to_pylist())
        if index % 100 == 0:
            print(f"Read {index}/{len(files)} parquet files", flush=True)

    dates = sorted(rows_by_date)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fields": ["race_id", "meet", "market", "combination", "odds", "is_hit", "is_capped_odds", "arrival_order"],
        "dates": [],
    }

    for race_date in dates:
        rows = sorted(
            rows_by_date[race_date],
            key=lambda row: (row[0] or "", MARKET_ORDER.get(row[2], 99), row[3]),
        )
        payload = {"date": race_date, "rows": rows}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        out_path = OUT_ROOT / f"{race_date}.json.gz"
        out_path.write_bytes(gzip.compress(body, compresslevel=9))
        manifest["dates"].append(
            {
                "date": race_date,
                "rows": len(rows),
                "file": f"data/odds/{race_date}.json.gz",
                "size_bytes": out_path.stat().st_size,
            }
        )

    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    total_size = sum((OUT_ROOT / f"{date}.json.gz").stat().st_size for date in dates)
    print(f"Wrote {len(dates)} date files to {OUT_ROOT.relative_to(ROOT)}", flush=True)
    print(f"Compressed odds data: {total_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
