#!/usr/bin/env python3
"""Serve the static browser and parquet-backed odds APIs."""

from __future__ import annotations

import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
PARSED_ROOT = ROOT / "KRA" / "parsed"
MARKETS = ["win", "place", "quinella", "exacta", "quinella_place", "trio", "trifecta"]
DEFAULT_LIMIT = 5000
MAX_LIMIT = 50000


def json_response(handler: SimpleHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler: SimpleHTTPRequestHandler, message: str, status: int = 400) -> None:
    json_response(handler, {"error": message}, status)


def market_files(market: str, race_date: str) -> list[Path]:
    year = race_date[:4]
    month = race_date[:7]
    pattern = f"kra_*/market={market}/year={year}/month={month}/*.parquet"
    return sorted(PARSED_ROOT.glob(pattern))


def available_dates() -> list[str]:
    dates = set()
    for path in PARSED_ROOT.glob("kra_*/market_status/year=*/month=*/*.parquet"):
        table = pq.read_table(path, columns=["race_date"])
        dates.update(value for value in table.column("race_date").to_pylist() if value)
    return sorted(dates)


DATE_CACHE = available_dates()


def normalize_row(row: dict, market: str) -> dict:
    combo = []
    for key in ("horse_no", "first_no", "second_no", "third_no"):
        value = row.get(key)
        if value is not None:
            combo.append(str(value))

    return {
        "race_date": row.get("race_date"),
        "race_id": row.get("race_id"),
        "meet": row.get("meet"),
        "market": market,
        "combination": "-".join(combo),
        "odds": row.get("odds"),
        "is_hit": row.get("is_hit"),
        "is_cancel": row.get("is_cancel"),
        "is_capped_odds": row.get("is_capped_odds"),
        "arrival_order": row.get("arrival_order"),
    }


def read_odds(race_date: str, market_filter: str, limit: int) -> dict:
    if race_date not in DATE_CACHE:
        return {"date": race_date, "markets": [], "races": [], "rows": [], "returned": 0, "total": 0, "truncated": False}

    markets = MARKETS if market_filter == "all" else [market_filter]
    rows = []
    total = 0

    for market in markets:
        for path in market_files(market, race_date):
            table = pq.read_table(path)
            mask = pc.equal(table["race_date"], race_date)
            filtered = table.filter(mask)
            total += filtered.num_rows
            if filtered.num_rows == 0 or len(rows) >= limit:
                continue
            remaining = limit - len(rows)
            for row in filtered.slice(0, remaining).to_pylist():
                rows.append(normalize_row(row, market))

    rows.sort(key=lambda item: (item["race_id"] or "", item["market"] or "", item["combination"] or ""))
    races = sorted({row["race_id"] for row in rows if row["race_id"]})
    return {
        "date": race_date,
        "markets": markets,
        "races": races,
        "rows": rows,
        "returned": len(rows),
        "total": total,
        "truncated": total > len(rows),
        "limit": limit,
    }


class DataBrowserHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean_path = parsed.path.lstrip("/") or "index.html"
        resolved = (DOCS_ROOT / clean_path).resolve()
        if not str(resolved).startswith(str(DOCS_ROOT.resolve())):
            return str(DOCS_ROOT / "index.html")
        return str(resolved)

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        parsed = urlparse(self.path)
        if parsed.path == "/api/dates":
            json_response(self, {"dates": DATE_CACHE})
            return

        if parsed.path == "/api/odds":
            query = parse_qs(parsed.query)
            race_date = query.get("date", [""])[0]
            market = query.get("market", ["all"])[0]
            try:
                limit = min(max(int(query.get("limit", [str(DEFAULT_LIMIT)])[0]), 1), MAX_LIMIT)
            except ValueError:
                limit = DEFAULT_LIMIT

            if not race_date:
                error_response(self, "Missing required query parameter: date")
                return
            if market != "all" and market not in MARKETS:
                error_response(self, f"Unsupported market: {market}")
                return

            json_response(self, read_odds(race_date, market, limit))
            return

        return super().do_GET()

    def end_headers(self) -> None:
        if self.path.endswith(".js"):
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
        super().end_headers()


def main() -> None:
    mimetypes.add_type("text/javascript", ".js")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DataBrowserHandler)
    print("Serving KRA data browser at http://127.0.0.1:8000", flush=True)
    print(f"Available dates: {DATE_CACHE[0]} through {DATE_CACHE[-1]} ({len(DATE_CACHE)} dates)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
