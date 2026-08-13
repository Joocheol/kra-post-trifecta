#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.turnover import extract_turnovers


MARKETS = [
    "win",
    "place",
    "quinella",
    "exacta",
    "quinella_place",
    "trio",
    "trifecta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--exclude-years", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-race-id", default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def race_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[:-8]
    return path.stem


def parse_race_id(race_id: str) -> tuple[str, str, int]:
    race_date, meet, race_no = race_id.split("_")
    return race_date, meet, int(race_no)


def rel_raw_key(path: Path, raw_root: Path) -> str:
    return str(path.relative_to(raw_root).as_posix())


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "nan":
        return ""
    return html.unescape(text).replace("\xa0", " ").strip()


def to_int(value: Any) -> int | None:
    text = clean_text(value)
    if not text or text == "-":
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group(0)) if m else None


def to_horse_no(value: Any) -> int | None:
    text = clean_text(value)
    return int(text) if re.fullmatch(r"\d+", text) else None


def to_odds(value: Any) -> float | None:
    text = clean_text(value).replace(",", "")
    if not text or text == "-":
        return None
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def is_capped(odds: float | None) -> bool:
    return odds is not None and odds >= 9999.9


def html_tables(page: str) -> list[pd.DataFrame]:
    if not page:
        return []
    try:
        return pd.read_html(StringIO(page), flavor="lxml")
    except Exception:
        return []


def extract_arrival_order(pages: dict[str, Any]) -> list[int]:
    for key in ("Scm", "Both", "Bc"):
        page = pages.get(key)
        if not isinstance(page, str):
            continue
        match = re.search(
            r'<label[^>]*arrival_mabun[^>]*>.*?</label>\s*<span>(.*?)</span>',
            page,
            flags=re.S,
        )
        if not match:
            match = re.search(r"도착마번.*?<span>(.*?)</span>", page, flags=re.S)
        if match:
            text = re.sub(r"<[^>]+>", " ", match.group(1))
            nums = [int(x) for x in re.findall(r"\d+", html.unescape(text))]
            if nums:
                return nums
    return []


def place_cutoff(n_valid: int) -> int:
    if n_valid >= 8:
        return 3
    if n_valid >= 5:
        return 2
    return 0


def last_col_label(col: Any) -> str:
    if isinstance(col, tuple):
        return clean_text(col[-1])
    return clean_text(col)


def table_matrix(table: pd.DataFrame) -> tuple[list[int], list[int], list[list[Any]]]:
    columns = [last_col_label(c) for c in table.columns]
    col_horses = [to_horse_no(c) for c in columns[1:]]
    row_horses: list[int] = []
    matrix: list[list[Any]] = []
    for _, row in table.iterrows():
        horse = to_horse_no(row.iloc[0])
        if horse is None:
            continue
        row_horses.append(horse)
        matrix.append(list(row.iloc[1:]))
    return row_horses, [c for c in col_horses if c is not None], matrix


def scm_components(table: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    win_rows: list[dict[str, Any]] = []
    place_rows: list[dict[str, Any]] = []
    quinella_rows: list[dict[str, Any]] = []
    columns = [last_col_label(c) for c in table.columns]
    quinella_cols = [to_horse_no(c) for c in columns[3:]]
    for _, row in table.iterrows():
        horse_no = to_horse_no(row.iloc[2])
        if horse_no is None:
            continue
        win_odds = to_odds(row.iloc[0])
        place_odds = to_odds(row.iloc[1])
        if win_odds is not None:
            win_rows.append({"horse_no": horse_no, "odds": win_odds})
        if place_odds is not None:
            place_rows.append({"horse_no": horse_no, "odds": place_odds})
        for col_no, value in zip(quinella_cols, list(row.iloc[3:])):
            if col_no is None or col_no <= horse_no:
                continue
            odds = to_odds(value)
            if odds is None:
                continue
            quinella_rows.append({"horse_a": horse_no, "horse_b": col_no, "odds": odds})
    return win_rows, place_rows, quinella_rows


def pair_matrix_rows(table: pd.DataFrame, ordered: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_horses, col_horses, matrix = table_matrix(table)
    for row_horse, values in zip(row_horses, matrix):
        for col_horse, value in zip(col_horses, values):
            if row_horse == col_horse:
                continue
            odds = to_odds(value)
            if odds is None:
                continue
            if ordered:
                rows.append({"first_no": col_horse, "second_no": row_horse, "odds": odds})
            elif row_horse < col_horse:
                rows.append({"horse_a": row_horse, "horse_b": col_horse, "odds": odds})
    return rows


def trio_rows(page_dict: dict[str, Any]) -> list[dict[str, Any]]:
    combos: dict[tuple[int, int, int], float] = {}
    for key, page in page_dict.items():
        if key == "_probe" or not isinstance(page, str):
            continue
        fixed = to_int(key)
        if fixed is None:
            continue
        tables = html_tables(page)
        if not tables:
            continue
        row_horses, col_horses, matrix = table_matrix(tables[0])
        for row_horse, values in zip(row_horses, matrix):
            for col_horse, value in zip(col_horses, values):
                if len({fixed, row_horse, col_horse}) != 3:
                    continue
                odds = to_odds(value)
                if odds is None:
                    continue
                combo = tuple(sorted((fixed, row_horse, col_horse)))
                combos.setdefault(combo, odds)
    return [
        {"horse_a": a, "horse_b": b, "horse_c": c, "odds": odds}
        for (a, b, c), odds in sorted(combos.items())
    ]


def trifecta_rows(page_dict: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for key, page in page_dict.items():
        if key == "_probe" or not isinstance(page, str):
            continue
        first = to_int(key)
        if first is None:
            continue
        tables = html_tables(page)
        if not tables:
            continue
        row_horses, col_horses, matrix = table_matrix(tables[0])
        for third, values in zip(row_horses, matrix):
            for second, value in zip(col_horses, values):
                if len({first, second, third}) != 3:
                    continue
                odds = to_odds(value)
                if odds is None:
                    continue
                key_tuple = (first, second, third)
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                rows.append({"first_no": first, "second_no": second, "third_no": third, "odds": odds})
    return rows


def with_common(row: dict[str, Any], race_id: str, race_date: str, meet: str, arrival: str) -> dict[str, Any]:
    row = dict(row)
    row.update({"race_id": race_id, "arrival_order": arrival, "race_date": race_date, "meet": meet})
    return row


@dataclass
class ParsedRace:
    valid_horses: dict[str, Any]
    markets: dict[str, list[dict[str, Any]]]
    market_status: list[dict[str, Any]]
    error: str = ""


def parse_raw(path: Path, raw_root: Path) -> ParsedRace:
    race_id = race_id_from_path(path)
    race_date, meet, race_no = parse_race_id(race_id)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    pages = payload.get("pages", {})
    meta = payload.get("meta", {})
    turnovers = extract_turnovers(pages)
    arrival_nums = extract_arrival_order(pages)
    arrival = ",".join(str(x) for x in arrival_nums)
    markets: dict[str, list[dict[str, Any]]] = {m: [] for m in MARKETS}

    scm_tables = html_tables(pages.get("Scm", "") if isinstance(pages.get("Scm"), str) else "")
    win_base: list[dict[str, Any]] = []
    place_base: list[dict[str, Any]] = []
    quinella_base: list[dict[str, Any]] = []
    if scm_tables:
        win_base, place_base, quinella_base = scm_components(scm_tables[0])
    valid_nums = sorted({row["horse_no"] for row in win_base} or set(arrival_nums))
    cutoff = place_cutoff(len(valid_nums))
    top2 = arrival_nums[:2]
    top_place = set(arrival_nums[:cutoff])
    top3 = arrival_nums[:3]
    if scm_tables:
        for row in win_base:
            horse = row["horse_no"]
            if valid_nums and horse not in valid_nums:
                continue
            row.update({"is_hit": bool(arrival_nums and horse == arrival_nums[0]), "is_capped_odds": is_capped(row["odds"])})
            markets["win"].append(with_common(row, race_id, race_date, meet, arrival))
        for row in place_base:
            horse = row["horse_no"]
            if valid_nums and horse not in valid_nums:
                continue
            row.update({"is_hit": bool(horse in top_place), "is_capped_odds": is_capped(row["odds"])})
            markets["place"].append(with_common(row, race_id, race_date, meet, arrival))
        for row in quinella_base:
            combo = {row["horse_a"], row["horse_b"]}
            row.update(
                {
                    "is_hit": bool(len(top2) == 2 and combo == set(top2)),
                    "is_cancel": False,
                    "is_capped_odds": is_capped(row["odds"]),
                }
            )
            markets["quinella"].append(with_common(row, race_id, race_date, meet, arrival))

    both_tables = html_tables(pages.get("Both", "") if isinstance(pages.get("Both"), str) else "")
    if both_tables:
        for row in pair_matrix_rows(both_tables[0], ordered=True):
            row.update(
                {
                    "is_hit": bool(len(top2) == 2 and row["first_no"] == top2[0] and row["second_no"] == top2[1]),
                    "is_cancel": False,
                    "is_capped_odds": is_capped(row["odds"]),
                }
            )
            markets["exacta"].append(with_common(row, race_id, race_date, meet, arrival))

    bc_tables = html_tables(pages.get("Bc", "") if isinstance(pages.get("Bc"), str) else "")
    if bc_tables:
        for row in pair_matrix_rows(bc_tables[0], ordered=False):
            combo = {row["horse_a"], row["horse_b"]}
            row.update(
                {
                    "is_hit": bool(cutoff and combo.issubset(top_place)),
                    "is_cancel": False,
                    "is_capped_odds": is_capped(row["odds"]),
                }
            )
            markets["quinella_place"].append(with_common(row, race_id, race_date, meet, arrival))

    if isinstance(pages.get("3Bc"), dict):
        for row in trio_rows(pages["3Bc"]):
            combo = {row["horse_a"], row["horse_b"], row["horse_c"]}
            row.update(
                {
                    "is_hit": bool(len(top3) == 3 and combo == set(top3)),
                    "is_cancel": False,
                    "is_capped_odds": is_capped(row["odds"]),
                }
            )
            markets["trio"].append(with_common(row, race_id, race_date, meet, arrival))

    if isinstance(pages.get("3Both"), dict):
        for row in trifecta_rows(pages["3Both"]):
            row.update(
                {
                    "is_hit": bool(
                        len(top3) == 3
                        and row["first_no"] == top3[0]
                        and row["second_no"] == top3[1]
                        and row["third_no"] == top3[2]
                    ),
                    "is_cancel": False,
                    "is_capped_odds": is_capped(row["odds"]),
                }
            )
            markets["trifecta"].append(with_common(row, race_id, race_date, meet, arrival))

    is_cancelled = not arrival_nums
    status_reason = "" if not is_cancelled else "missing_arrival_order"
    raw_key = rel_raw_key(path, raw_root)
    valid_row = {
        "race_id": race_id,
        "race_date": race_date,
        "meet": meet,
        "race_no": race_no,
        "n_valid_horses": len(valid_nums),
        "valid_horses": ",".join(str(x) for x in valid_nums),
        "arrival_order": arrival,
        "arrival_len": len(arrival_nums),
        "place_cutoff": cutoff,
        "is_canceled_or_no_result": is_cancelled,
        "race_status_reason": status_reason,
        "raw_s3_key": raw_key,
    }
    status_rows: list[dict[str, Any]] = []
    for market in MARKETS:
        n_rows = len(markets[market])
        status_rows.append(
            {
                "race_id": race_id,
                "market": market,
                "n_rows": n_rows,
                "status": "ok" if n_rows else "no_rows",
                "is_cancelled": is_cancelled,
                "status_reason": "parsed_rows_present" if n_rows else "no_parsed_rows",
                "turnover_won": turnovers[market],
                "race_date": race_date,
                "meet": meet,
            }
        )
    return ParsedRace(valid_row, markets, status_rows)


SCHEMAS = {
    "valid_horses": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
            ("race_no", pa.int64()),
            ("n_valid_horses", pa.int64()),
            ("valid_horses", pa.large_string()),
            ("arrival_order", pa.large_string()),
            ("arrival_len", pa.int64()),
            ("place_cutoff", pa.int64()),
            ("is_canceled_or_no_result", pa.bool_()),
            ("race_status_reason", pa.large_string()),
            ("raw_s3_key", pa.large_string()),
        ]
    ),
    "market_status": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("market", pa.large_string()),
            ("n_rows", pa.int64()),
            ("status", pa.large_string()),
            ("is_cancelled", pa.bool_()),
            ("status_reason", pa.large_string()),
            ("turnover_won", pa.int64()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "win": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("horse_no", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "place": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("horse_no", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "quinella": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("horse_a", pa.int64()),
            ("horse_b", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("is_cancel", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "quinella_place": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("horse_a", pa.int64()),
            ("horse_b", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("is_cancel", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "exacta": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("first_no", pa.int64()),
            ("second_no", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("is_cancel", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "trio": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("horse_a", pa.int64()),
            ("horse_b", pa.int64()),
            ("horse_c", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("is_cancel", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
    "trifecta": pa.schema(
        [
            ("race_id", pa.large_string()),
            ("first_no", pa.int64()),
            ("second_no", pa.int64()),
            ("third_no", pa.int64()),
            ("odds", pa.float64()),
            ("is_hit", pa.bool_()),
            ("arrival_order", pa.large_string()),
            ("is_cancel", pa.bool_()),
            ("is_capped_odds", pa.bool_()),
            ("race_date", pa.large_string()),
            ("meet", pa.large_string()),
        ]
    ),
}


def write_partition(out_root: Path, year_label: str, month: str, kind: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    if kind in MARKETS:
        rel = Path(year_label) / f"market={kind}" / f"year={month[:4]}" / f"month={month}" / "part-0.parquet"
    else:
        rel = Path(year_label) / kind / f"year={month[:4]}" / f"month={month}" / "part-0.parquet"
    path = out_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMAS[kind])
    pq.write_table(table, path, compression="snappy")
    return len(rows)


def year_label_for(race_date: str) -> str:
    if race_date.startswith("2016") and race_date >= "2016-06-10":
        return "kra_2016_post"
    return f"kra_{race_date[:4]}"


def wanted(path: Path, args: argparse.Namespace) -> bool:
    rid = race_id_from_path(path)
    try:
        race_date, _, _ = parse_race_id(rid)
    except Exception:
        return False
    if args.start_date and race_date < args.start_date:
        return False
    if args.end_date and race_date > args.end_date:
        return False
    exclude_years = {x.strip() for x in args.exclude_years.split(",") if x.strip()}
    if race_date[:4] in exclude_years:
        return False
    if args.sample_race_id and rid != args.sample_race_id:
        return False
    return True


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    out_root = args.out_root
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_root.glob("kra_*/raw_archive/*/*/*.json.gz"))
    files = [p for p in files if wanted(p, args)]
    if args.limit:
        files = files[: args.limit]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_buffer: list[dict[str, Any]] = []
    valid_buffer: list[dict[str, Any]] = []
    current_month_key: tuple[str, str] | None = None
    errors: list[dict[str, str]] = []
    race_counts = Counter()
    row_counts: list[dict[str, Any]] = []

    def flush_month() -> None:
        nonlocal grouped, status_buffer, valid_buffer, current_month_key
        if current_month_key is None:
            return
        year_label, month = current_month_key
        n = write_partition(out_root, year_label, month, "valid_horses", valid_buffer)
        row_counts.append({"year_label": year_label, "month": month, "kind": "valid_horses", "rows": n})
        n = write_partition(out_root, year_label, month, "market_status", status_buffer)
        row_counts.append({"year_label": year_label, "month": month, "kind": "market_status", "rows": n})
        for market in MARKETS:
            n = write_partition(out_root, year_label, month, market, grouped.get(market, []))
            row_counts.append({"year_label": year_label, "month": month, "kind": market, "rows": n})
        grouped = defaultdict(list)
        status_buffer = []
        valid_buffer = []

    for idx, path in enumerate(files, 1):
        rid = race_id_from_path(path)
        try:
            parsed = parse_raw(path, raw_root)
            race_date, _, _ = parse_race_id(rid)
            year_label = year_label_for(race_date)
            month = race_date[:7]
            month_key = (year_label, month)
            if current_month_key is None:
                current_month_key = month_key
            elif month_key != current_month_key:
                flush_month()
                current_month_key = month_key
            valid_buffer.append(parsed.valid_horses)
            status_buffer.extend(parsed.market_status)
            for market, rows in parsed.markets.items():
                grouped[market].extend(rows)
                race_counts[(market, len(rows))] += 1
        except Exception as exc:
            errors.append({"race_id": rid, "path": str(path), "error": repr(exc)})
        if idx % 500 == 0:
            print(f"parsed {idx}/{len(files)} errors={len(errors)}", flush=True)
    flush_month()

    manifest = args.manifest or out_root / "reparse_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "raw_root": str(raw_root),
                "out_root": str(out_root),
                "input_files": len(files),
                "errors": len(errors),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "exclude_years": args.exclude_years,
                "row_counts": row_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        err_path = out_root / "parse_errors.csv"
        with err_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["race_id", "path", "error"])
            writer.writeheader()
            writer.writerows(errors)
    print(f"done input_files={len(files)} errors={len(errors)} out={out_root} manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
