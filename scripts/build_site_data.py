#!/usr/bin/env python3
"""Build compact metadata for the static data browser."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "KRA" / "raw_collected_v3_15w"
PARSED_ROOT = ROOT / "KRA" / "parsed"
OUT_PATH = ROOT / "docs" / "data" / "site-summary.json"


def bytes_to_mb(size: int) -> float:
    return round(size / 1024 / 1024, 2)


def top_level_dataset(path: Path) -> str:
    try:
        return path.relative_to(PARSED_ROOT).parts[0]
    except ValueError:
        return "unknown"


def parsed_kind(path: Path) -> str:
    rel_parts = path.relative_to(PARSED_ROOT).parts
    for part in rel_parts:
        if part.startswith("market="):
            return part.split("=", 1)[1]
        if part in {"market_status", "valid_horses"}:
            return part
    return "unknown"


def parsed_year(path: Path) -> str:
    rel_parts = path.relative_to(PARSED_ROOT).parts
    for part in rel_parts:
        if part.startswith("year="):
            return part.split("=", 1)[1]
    dataset = top_level_dataset(path)
    if dataset.startswith("kra_") and dataset[4:8].isdigit():
        return dataset[4:8]
    return "unknown"


def parsed_month(path: Path) -> str:
    rel_parts = path.relative_to(PARSED_ROOT).parts
    for part in rel_parts:
        if part.startswith("month="):
            return part.split("=", 1)[1]
    return "unknown"


def summarize_raw() -> dict:
    files = sorted(RAW_ROOT.glob("kra_*/raw_archive/*/*/*.json.gz"))
    by_year: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    dates: list[str] = []
    total_bytes = 0

    for path in files:
        race_date = path.name[:10]
        dates.append(race_date)
        by_year[race_date[:4]] += 1
        by_month[race_date[:7]] += 1
        by_dataset[path.relative_to(RAW_ROOT).parts[0]] += 1
        total_bytes += path.stat().st_size

    samples = []
    for path in [files[0], files[len(files) // 2], files[-1]] if files else []:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        samples.append(
            {
                "path": str(path.relative_to(ROOT)),
                "race_date": path.name[:10],
                "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
                "json_type": type(payload).__name__,
            }
        )

    return {
        "file_count": len(files),
        "size_mb": bytes_to_mb(total_bytes),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "by_year": dict(sorted(by_year.items())),
        "by_month": dict(sorted(by_month.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "samples": samples,
    }


def summarize_parsed() -> dict:
    files = sorted(PARSED_ROOT.glob("**/*.parquet"))
    by_year = defaultdict(lambda: {"files": 0, "rows": 0, "size_mb": 0.0})
    by_kind = defaultdict(lambda: {"files": 0, "rows": 0, "size_mb": 0.0})
    by_dataset = defaultdict(lambda: {"files": 0, "rows": 0, "size_mb": 0.0})
    by_month: Counter[str] = Counter()
    schemas: dict[str, list[str]] = {}
    total_rows = 0
    total_bytes = 0
    errors = []

    for path in files:
        size = path.stat().st_size
        total_bytes += size
        kind = parsed_kind(path)
        year = parsed_year(path)
        month = parsed_month(path)
        dataset = top_level_dataset(path)

        try:
            parquet_file = pq.ParquetFile(path)
            rows = parquet_file.metadata.num_rows
            if kind not in schemas:
                schemas[kind] = list(parquet_file.schema_arrow.names)
        except Exception as exc:  # pragma: no cover - diagnostic path
            rows = 0
            errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})

        total_rows += rows
        by_year[year]["files"] += 1
        by_year[year]["rows"] += rows
        by_year[year]["size_mb"] += size / 1024 / 1024
        by_kind[kind]["files"] += 1
        by_kind[kind]["rows"] += rows
        by_kind[kind]["size_mb"] += size / 1024 / 1024
        by_dataset[dataset]["files"] += 1
        by_dataset[dataset]["rows"] += rows
        by_dataset[dataset]["size_mb"] += size / 1024 / 1024
        by_month[month] += 1

    def finalize(mapping: dict) -> dict:
        finalized = {}
        for key, value in sorted(mapping.items()):
            finalized[key] = {
                "files": value["files"],
                "rows": value["rows"],
                "size_mb": round(value["size_mb"], 2),
            }
        return finalized

    return {
        "file_count": len(files),
        "row_count": total_rows,
        "size_mb": bytes_to_mb(total_bytes),
        "by_year": finalize(by_year),
        "by_kind": finalize(by_kind),
        "by_dataset": finalize(by_dataset),
        "by_month": dict(sorted(by_month.items())),
        "schemas": schemas,
        "errors": errors,
    }


def main() -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": "kra-post-trifecta-analysis",
        "scope": {
            "start_date": "2016-06-10",
            "end_date": "2025-12-31",
            "excluded_years": ["2020", "2021"],
            "reason": "Post-trifecta market era",
        },
        "raw": summarize_raw(),
        "parsed": summarize_parsed(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
