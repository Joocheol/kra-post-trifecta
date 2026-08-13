"""Extract race-market turnover from archived KRA result pages."""

from __future__ import annotations

import html
import re
from typing import Any


TURNOVER_PAGE_LABELS = {
    "win": ("Scm", "단승식"),
    "place": ("Scm", "연승식"),
    "quinella": ("Scm", "복승식"),
    "exacta": ("Both", "쌍승식"),
    "quinella_place": ("Bc", "복연승식"),
    "trio": ("3Bc", "삼복승식"),
    "trifecta": ("3Both", "삼쌍승식"),
}


def _page_text(page: str) -> str:
    text = re.sub(r"<[^>]+>", " ", page)
    return " ".join(html.unescape(text).replace("\xa0", " ").split())


def _representative_page(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    probe = value.get("_probe")
    if isinstance(probe, str):
        return probe
    for key in sorted(value, key=str):
        page = value[key]
        if isinstance(page, str):
            return page
    return ""


def _labelled_turnover(page: str, label: str, combined_page: bool) -> int | None:
    text = _page_text(page)
    if combined_page:
        pattern = rf"(?<![가-힣]){re.escape(label)}\s*:\s*([0-9,]+)\s*원"
    else:
        pattern = rf"(?<![가-힣]){re.escape(label)}\s*매출총액\s*:?\s*([0-9,]+)\s*원"
    match = re.search(pattern, text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def extract_turnovers(pages: dict[str, Any]) -> dict[str, int | None]:
    """Return posted turnover in won for every KRA wagering market.

    The ``Scm`` page reports win, place, and quinella turnover separately in
    its footer. Other market pages report one labelled market total. Paginated
    trio and trifecta pages repeat the same total, so one representative page
    is sufficient.
    """

    result: dict[str, int | None] = {}
    for market, (page_key, label) in TURNOVER_PAGE_LABELS.items():
        page = _representative_page(pages.get(page_key))
        result[market] = _labelled_turnover(
            page, label, combined_page=page_key == "Scm"
        )
    return result
